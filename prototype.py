import os
import io
import re
import json
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- LLM: Gemini 공식 SDK ---
import google.generativeai as genai

# --------- 기본 세팅 ----------
st.set_page_config(page_title="HARU", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: blue; font-size: 80px;'>H A R U</h1>",
    unsafe_allow_html=True
)

# --------- Gemini 설정 ----------
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.warning("환경변수 GOOGLE_API_KEY가 비어있어요. 키를 설정해 주세요.")
genai.configure(api_key=API_KEY)

# 지원 파일 형식 (CSV/TXT 추가)
ALLOWED_EXTS = {".xlsx", ".xls", ".csv", ".docx", ".pdf", ".txt"}

# ================== 유틸/파서 ==================
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
        return io.BytesIO(file_bytes).getvalue().decode("cp949", errors="ignore")

def get_extension(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", filename)
    return m.group(0).lower() if m else ""

# --------- 텍스트 EDA ----------
from collections import Counter

def text_eda_summary(text: str) -> dict:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
    return {
        "chars": len(text),
        "words": len(tokens),
        "unique_words": len(set(t.lower() for t in tokens)),
        "top_terms": Counter(t.lower() for t in tokens if len(t) >= 2).most_common(20),
    }

def render_text_eda(summary: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("문자 수", f"{summary['chars']:,}")
    c2.metric("단어 수", f"{summary['words']:,}")
    c3.metric("고유 단어 수", f"{summary['unique_words']:,}")
    st.write("상위 단어(상위 20)")
    st.dataframe(pd.DataFrame(summary["top_terms"], columns=["term", "freq"]), use_container_width=True)

# ================== EDA ==================

def dataframe_preview_metrics(df: pd.DataFrame):
    """표 형태 미리보기(head)를 없애고, 메트릭/리스트로만 요약"""
    shape = df.shape
    with st.container():
        st.markdown("#### 📌 데이터 미리보기(요약)")
        c1, c2, c3 = st.columns(3)
        c1.metric("행(Row)", f"{shape[0]:,}")
        c2.metric("열(Column)", f"{shape[1]:,}")
        c3.metric("결측치 포함 열 수", f"{int((df.isna().any()).sum()):,}")

        st.markdown("**컬럼 목록(상위 20개까지)**")
        st.write(", ".join(list(df.columns)[:20]))

def dataframe_eda_retail(df: pd.DataFrame):
    """EDA"""
    # 3-1) 결측치/기본 통계
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    st.subheader("결측치 비율(%)")
    st.dataframe(missing.to_frame("missing_%"), use_container_width=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.subheader("수치형 기술통계")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)


    # 3-2) 빠른 시각화(간단)
    if num_cols:
        with st.expander("📊 빠른 시각화"):
            target = st.selectbox("히스토그램 대상", num_cols, key="hist_col")
            st.plotly_chart(px.histogram(df, x=target), use_container_width=True)
            if len(num_cols) >= 2:
                xcol = st.selectbox("산점도 X", num_cols, key="xcol")
                ycol = st.selectbox("산점도 Y", num_cols, key="ycol")
                st.plotly_chart(px.scatter(df, x=xcol, y=ycol), use_container_width=True)

    # 3-3) 로컬 텍스트 설명
    st.subheader("📄 로컬 EDA 설명")
    st.text(summarize_eda_local(df))

def summarize_eda_local(df: pd.DataFrame) -> str:
    total_rows, total_cols = df.shape
    missing_ratio = (df.isna().mean() * 100).round(2)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    desc = df[num_cols].describe().T.round(2) if num_cols else None

    lines = []
    lines.append(f"총 행 수: {total_rows} / 열 수: {total_cols}")
    if not missing_ratio.empty:
        lines.append("결측치 비율 상위 5개 열:")
        lines.extend([f"- {col}: {missing_ratio[col]}%" for col in missing_ratio.head(5).index])
    if desc is not None and not desc.empty:
        lines.append("변동성이 큰 열 Top 3:")
        var_top = desc.sort_values("std", ascending=False).head(3).index.tolist()
        lines.extend([f"- {col}" for col in var_top])
    return "\n".join(lines)

# ================== 이상탐지 ==================
def detect_anomalies(df: pd.DataFrame, cols: List[str], contamination: float = 0.03, n_estimators: int = 300) -> Optional[pd.DataFrame]:
    if not cols:
        st.warning("이상탐지에 사용할 열을 선택해 주세요.")
        return None
    sub = df[cols].dropna()
    if sub.shape[0] < 10:
        st.warning("이상탐지에 충분한 행이 없습니다(최소 10행 권장).")
        return None
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    labels = iso.fit_predict(sub.values)  # -1: anomaly, 1: normal
    scores = iso.score_samples(sub.values)  # 점수(작을수록 이상)
    res = df.copy()
    res["_anomaly"] = "normal"
    res.loc[sub.index[labels == -1], "_anomaly"] = "anomaly"
    res.loc[sub.index, "_anomaly_score"] = scores
    return res

def summarize_anomalies_local(res: pd.DataFrame, used_cols: List[str]) -> str:
    if "_anomaly" not in res:
        return "이상탐지 결과가 없습니다."
    total = len(res)
    num_anom = (res["_anomaly"] == "anomaly").sum()
    ratio = round(num_anom / total * 100, 2)
    lines = [f"전체 {total}행 중 이상치 {num_anom}개 ({ratio}%) 발견",
             f"사용 열: {', '.join(used_cols) if used_cols else '(선택 없음)'}"]
    anoms = res[res["_anomaly"]=="anomaly"].copy()
    if not anoms.empty:
        if "_anomaly_score" in anoms:
            anoms = anoms.sort_values("_anomaly_score").head(5)
        else:
            anoms = anoms.head(5)
        show_cols = ["_anomaly_score"] + [c for c in used_cols if c in res.columns][:5]
        lines.append("상위 이상치 미리보기:")
        lines.append(str(anoms[show_cols].round(4)))
    return "\n".join(lines)

# ================== 요약 JSON(질문용) ==================
def build_summary_for_llm(df: pd.DataFrame, res: Optional[pd.DataFrame], used_cols: Optional[List[str]] = None) -> str:
    summary = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": list(df.columns),
        "missing_ratio": (df.isna().mean() * 100).round(2).to_dict(),
    }
    num = df.select_dtypes(include="number")
    if not num.empty:
        summary["describe"] = num.describe().T.round(2).to_dict()
        corr = num.corr(numeric_only=True).abs().unstack().sort_values(ascending=False)
        corr = corr[corr.index.get_level_values(0) < corr.index.get_level_values(1)]
        summary["top_corr_pairs"] = [{"pair": f"{a}~{b}", "corr": round(float(v),4)} for (a,b), v in corr.head(8).items()]
    if isinstance(res, pd.DataFrame) and "_anomaly" in res:
        total = len(res)
        num_anom = int((res["_anomaly"] == "anomaly").sum())
        summary["anomaly"] = {
            "count": num_anom,
            "ratio_pct": round((num_anom / total * 100) if total else 0.0, 2),
            "used_cols": used_cols or num.columns.tolist()
        }
        anoms = res[res["_anomaly"] == "anomaly"].copy()
        if "_anomaly_score" in anoms:
            anoms = anoms.sort_values("_anomaly_score").head(20)
        else:
            anoms = anoms.head(20)
        num_cols = num.columns.tolist()
        cols_show = (["_anomaly_score"] if "_anomaly_score" in anoms.columns else []) + (used_cols or num_cols)[:6]
        summary["top_anomalies_preview"] = anoms[cols_show].round(4).reset_index().to_dict(orient="records")
    return json.dumps(summary, ensure_ascii=False)

def build_context_from_file(ext: str, df: Optional[pd.DataFrame], text: Optional[str]) -> str:
    if ext in {".docx", ".pdf", ".txt"} and text is not None:
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

# 2) 파일 삽입
st.subheader("2) 파일 삽입 (Excel / CSV / Word / PDF / TXT)")
up = st.file_uploader("Upload your file first", type=list({e.strip('.') for e in ALLOWED_EXTS}))

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
            st.success("Word 파일 로드 성공")
        elif ext == ".pdf":
            doc_text = read_pdf(data)
            st.success("PDF 파일 로드 성공")
        elif ext == ".txt":
            doc_text = read_txt(data)
            st.success("TXT 파일 로드 성공")
    except Exception as e:
        st.error(f"파일 파싱 실패: {e}")

    # 2) 추출 텍스트 미리보기
    st.subheader("📄 추출 텍스트 미리보기")
    if df is not None:
        schema = {c: str(df[c].dtype) for c in df.columns}
        preview = {"columns": df.columns.tolist(), "dtypes": schema}
        st.text_area("테이블 스키마(JSON)", json.dumps(preview, ensure_ascii=False, indent=2), height=200)
        # 2-1) 데이터 미리보기(메트릭형) — 표 head 제거
        dataframe_preview_metrics(df)
        # 세션 보존
        st.session_state["df"] = df
    elif doc_text is not None:
        st.text_area("문서 본문 일부", doc_text[:5000], height=200)
        render_text_eda(text_eda_summary(doc_text))
        st.session_state["doc_text"] = doc_text
    else:
        st.info("텍스트를 추출할 수 없습니다. 파일 형식을 확인해주세요.")

st.divider()

# 3) EDA
st.subheader("3) EDA")
df = st.session_state.get("df", df)
doc_text = st.session_state.get("doc_text", doc_text)
if df is not None:
    dataframe_eda_retail(df)
elif doc_text is not None:
    st.info("문서형 파일은 상단 '추출 텍스트 미리보기'에서 EDA 지표를 확인하세요.")
else:
    st.info("Upload your file first")

st.divider()

# 4) 이상분석 (토글 없이 명시 버튼)
st.subheader("4) 이상분석")
if df is not None:
    num_cols_all = df.select_dtypes(include="number").columns.tolist()
    if num_cols_all:
        c1, c2 = st.columns([2, 1])
        with c1:
            sel_cols = st.multiselect("이상탐지에 사용할 수치형 열을 선택하세요", num_cols_all, key="anom_cols_sel")
        with c2:
            contamination = st.slider("오염도(이상 비율 추정)", 0.01, 0.2, 0.03, 0.01, key="anom_cont")
        run = st.button("🚨 이상탐지 실행", use_container_width=True)
        if run:
            res = detect_anomalies(df, sel_cols, contamination=contamination)
            if res is not None:
                st.session_state["res"] = res
                st.session_state["anom_cols"] = sel_cols
                st.success(f"이상탐지 완료: 이상치 {(res['_anomaly']=='anomaly').sum()}건")
                # 결과 테이블 및 간단 시각화
                st.dataframe(res, use_container_width=True)
                if len(sel_cols) == 1:
                    tmp = res[[sel_cols[0], "_anomaly"]].copy()
                    tmp["_idx"] = range(len(tmp))
                    st.plotly_chart(px.scatter(tmp, x="_idx", y=sel_cols[0], color="_anomaly"), use_container_width=True)
                # 로컬 텍스트 설명
                st.markdown("### 📄 로컬 이상탐지 설명")
                st.text(summarize_anomalies_local(res, sel_cols))
    else:
        st.info("수치형 열이 없습니다. 이상탐지를 건너뜁니다.")
elif doc_text is not None:
    st.info("문서형 파일은 이상탐지를 지원하지 않습니다.")
else:
    st.info("Upload your file first")

st.divider()

# 5) 질문하기 (LLM 응답) — 요약본만 전송
st.subheader("5) 질문하기 (LLM 응답)")
user_q = st.text_input("질문을 입력하세요", placeholder="Ask a Question here")
if st.button("질문 전송", use_container_width=True) and user_q.strip():
    df = st.session_state.get("df", df)
    res = st.session_state.get("res", None)
    used_cols = st.session_state.get("anom_cols", None)

    if df is not None:
        summary_json = build_summary_for_llm(df, res, used_cols)
        system_prompt = (
            "너는 업로드된 데이터 요약본(JSON)을 근거로 한국어로 간결하고 근거 있는 분석을 제공하는 전문가야. "
            "JSON의 수치와 통계를 근거로 사용자 질문에 답해. JSON 밖 정보는 추측하지 말고 모른다고 답해."
        )
        user_prompt = f"[요약 JSON]\n{summary_json}\n\n[질문]\n{user_q}"
    elif doc_text is not None:
        ctx = build_context_from_file(ext or "", None, doc_text)
        system_prompt = "너는 문서 내용을 바탕으로 한국어로 답변하는 전문가야. 컨텍스트 밖은 추측하지 마."
        user_prompt = f"[컨텍스트]\n{ctx}\n\n[질문]\n{user_q}"
    else:
        st.warning("Upload your file first")
        st.stop()

    with st.spinner("생각 중..."):
        answer = ask_llm(model_name, system_prompt, user_prompt, temperature=temperature)
    st.markdown("### 🧠 LLM 응답")
    st.markdown(answer)
