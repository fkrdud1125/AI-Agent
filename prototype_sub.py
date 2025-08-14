import os
import io
import re
import json
from typing import Optional, List
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- LLM: Gemini 공식 SDK ---
import google.generativeai as genai

# --------- 기본 세팅 ----------
st.set_page_config(page_title="HARU AI Portal", layout="wide")
st.title("🤖 HARU AI Portal (Gemini)")

# --------- Gemini 설정 ----------
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.warning("환경변수 GOOGLE_API_KEY가 비어있어요. 키를 설정해 주세요.")
genai.configure(api_key=API_KEY)

# 지원 파일 형식 (CSV/TXT 추가)
ALLOWED_EXTS = {".xlsx", ".xls", ".csv", ".docx", ".pdf", ".txt"}

# --------- LLM 유틸 ----------
def get_model(model_name: str, temperature: float = 0.2):
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": temperature}
    )

def ask_llm(model_name: str, system_msg: str, user_msg: str, temperature: float = 0.2) -> str:
    model = get_model(model_name, temperature)
    parts = []
    if system_msg:
        parts.append(f"[SYSTEM]\n{system_msg}")
    parts.append(f"[USER]\n{user_msg}")
    resp = model.generate_content("\n\n".join(parts))
    try:
        return resp.text
    except Exception:
        return "(응답을 생성하지 못했습니다.)"

# --------- 파일 파서 ----------
def read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))

def read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def read_word(file_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def read_txt(file_bytes: bytes, encoding: str = "utf-8") -> str:
    try:
        return io.BytesIO(file_bytes).getvalue().decode(encoding)
    except UnicodeDecodeError:
        # 한글 파일에서 간혹 cp949 등 필요할 때
        return io.BytesIO(file_bytes).getvalue().decode("cp949", errors="ignore")

def get_extension(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", filename)
    return m.group(0).lower() if m else ""

# --------- 텍스트 EDA ----------
from collections import Counter
def text_eda_summary(text: str) -> dict:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    word_count = len(tokens)
    char_count = len(text)
    unique_count = len(set(t.lower() for t in tokens))
    top_terms = Counter(t.lower() for t in tokens if len(t) >= 2).most_common(20)
    return {"chars": char_count, "words": word_count, "unique_words": unique_count, "top_terms": top_terms}

def render_text_eda(summary: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("문자 수", f"{summary['chars']:,}")
    c2.metric("단어 수", f"{summary['words']:,}")
    c3.metric("고유 단어 수", f"{summary['unique_words']:,}")
    st.write("상위 단어(상위 20)")
    st.dataframe(pd.DataFrame(summary["top_terms"], columns=["term", "freq"]), use_container_width=True)

# --------- 데이터프레임 EDA ----------
def dataframe_eda(df: pd.DataFrame):
    st.write("샘플 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("결측치 비율(%)")
    missing = (df.isna().mean() * 100).round(2)
    st.dataframe(missing.to_frame("missing_%"), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.write("수치형 기술통계")
        st.dataframe(df.describe().T, use_container_width=True)

        with st.expander("📊 빠른 시각화"):
            target = st.selectbox("히스토그램 대상", numeric_cols, key="hist_col")
            fig = px.histogram(df, x=target)
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) >= 2:
                xcol = st.selectbox("산점도 X", numeric_cols, key="xcol")
                ycol = st.selectbox("산점도 Y", numeric_cols, key="ycol")
                fig2 = px.scatter(df, x=xcol, y=ycol)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("수치형 열이 없습니다.")

# --------- 이상탐지 ----------
def detect_anomalies(df: pd.DataFrame, cols: List[str], contamination: float = 0.03, n_estimators: int = 300) -> Optional[pd.DataFrame]:
    if not cols:
        st.warning("이상탐지에 사용할 수치형 열을 선택해 주세요.")
        return None
    sub = df[cols].dropna()
    if sub.shape[0] < 10:
        st.warning("이상탐지에 충분한 행이 없습니다(최소 10행 권장).")
        return None
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    lab = iso.fit_predict(sub.values)  # -1: anomaly, 1: normal
    res = df.copy()
    res["_anomaly"] = "normal"
    res.loc[sub.index[lab == -1], "_anomaly"] = "anomaly"
    return res

# --------- LLM 컨텍스트 ----------
def build_context_from_file(ext: str, df: Optional[pd.DataFrame], text: Optional[str]) -> str:
    """
    질문 시 LLM에 넘길 컨텍스트(요약 스냅샷).
    - Excel/CSV: 칼럼, dtype, head(5)
    - Word/PDF/TXT: 본문 일부 + 상위 단어
    """
    if ext in {".xlsx", ".xls", ".csv"} and df is not None:
        snap = {
            "columns": df.columns.tolist(),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "head": df.head(5).to_dict(orient="records")
        }
        return "다음은 업로드된 테이블 데이터(Excel/CSV)의 스냅샷입니다.\n" + json.dumps(snap, ensure_ascii=False)
    elif ext in {".docx", ".pdf", ".txt"} and text is not None:
        s = text.strip()
        s_short = s[:4000]
        eda = text_eda_summary(s)
        top = ", ".join([f"{t}({c})" for t, c in eda["top_terms"][:10]])
        return f"다음은 업로드된 문서의 일부 텍스트입니다:\n{s_short}\n\n상위 단어: {top}\n"
    else:
        return "컨텍스트가 없습니다."

# ================== UI ==================

# 1) LLM 모델 선택창
st.subheader("1) LLM 모델 선택")
model_name = st.selectbox("Gemini 모델", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

st.divider()

# 2) 파일 삽입 (CSV/TXT 추가)
st.subheader("2) 파일 삽입 (Excel / CSV / Word / PDF / TXT)")
up = st.file_uploader("파일을 업로드하세요", type=list({e.strip('.') for e in ALLOWED_EXTS}))

ext = None
df: Optional[pd.DataFrame] = None
doc_text: Optional[str] = None

if up is not None:
    ext = get_extension(up.name)
    if ext not in ALLOWED_EXTS:
        st.error("옳지 않은 파일 형식입니다. Excel(.xlsx/.xls), CSV(.csv), Word(.docx), PDF(.pdf), TXT(.txt)만 지원해요.")
        st.stop()

    data = up.read()

    try:
        if ext in {".xlsx", ".xls"}:
            df = read_excel(data)
            st.success(f"엑셀 파일 로드 성공: {df.shape[0]}행 × {df.shape[1]}열")
        elif ext == ".csv":
            df = read_csv(data)
            st.success(f"CSV 파일 로드 성공: {df.shape[0]}행 × {df.shape[1]}열")
        elif ext == ".docx":
            doc_text = read_word(data)
            st.success("워드 파일 로드 성공")
        elif ext == ".pdf":
            doc_text = read_pdf(data)
            st.success("PDF 파일 로드 성공")
        elif ext == ".txt":
            doc_text = read_txt(data)
            st.success("TXT 파일 로드 성공")
    except Exception as e:
        st.error(f"파일 파싱 실패: {e}")

    # ===== (신규) 업로드 직후 텍스트 추출 미리보기 =====
    st.subheader("📄 추출 텍스트 미리보기")
    if df is not None:
        # 표 데이터를 텍스트로도 보여주기: 스키마 + head를 JSON 문자열로
        schema = {c: str(df[c].dtype) for c in df.columns}
        preview = {
            "columns": df.columns.tolist(),
            "dtypes": schema,
            "head": df.head(5).to_dict(orient="records"),
        }
        st.text_area("테이블 스냅샷(JSON)", json.dumps(preview, ensure_ascii=False, indent=2), height=220)
        st.caption("※ 아래에서 EDA/이상탐지를 계속 진행하세요.")
    elif doc_text is not None:
        st.text_area("문서 본문 일부", doc_text[:5000], height=220)
        st.caption("※ 아래에서 텍스트 EDA를 계속 진행하세요.")
    else:
        st.info("텍스트를 추출할 수 없습니다. 파일 형식을 확인해주세요.")

st.divider()

# 3) 파일 EDA와 이상분석
st.subheader("3) 파일 EDA와 이상분석")
if df is not None:
    dataframe_eda(df)
    with st.expander("🚨 이상탐지 (IsolationForest)"):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            sel = st.multiselect("이상탐지에 사용할 열", num_cols)
            contamination = st.slider("오염도(이상 비율 추정)", 0.01, 0.2, 0.03, 0.01)
            if st.button("이상탐지 실행"):
                res = detect_anomalies(df, sel, contamination=contamination)
                if res is not None:
                    st.dataframe(res, use_container_width=True)
                    if len(sel) == 1:
                        tmp = res[[sel[0], "_anomaly"]].copy()
                        tmp["_idx"] = range(len(tmp))
                        fig = px.scatter(tmp, x="_idx", y=sel[0], color="_anomaly")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("수치형 열이 없습니다. 이상탐지를 건너뜁니다.")
elif doc_text is not None:
    st.write("문서 EDA (텍스트 기반)")
    eda = text_eda_summary(doc_text)
    render_text_eda(eda)
else:
    st.info("먼저 파일을 업로드하세요. (Excel/CSV/Word/PDF/TXT)")

st.divider()

# 4) 질문하기 (LLM 응답)
st.subheader("4) 질문하기 (LLM 응답)")
user_q = st.text_input("질문을 입력하세요", ...)
if st.button("질문 전송", use_container_width=True) and user_q.strip():
    ctx = build_context_from_file(ext or "", df, doc_text)
    system_prompt = (...)
    user_prompt = f"[컨텍스트]\n{ctx}\n\n[질문]\n{user_q}"
    with st.spinner("생각 중..."):
        answer = ask_llm(...)
