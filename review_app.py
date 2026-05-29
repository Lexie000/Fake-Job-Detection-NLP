"""
Two-Stage Fraud Job Detection — Review App
==========================================
Stage 1: Hybrid SVM (TF-IDF + metadata)  →  fast fraud probability
Stage 2: LLM evidence extractor          →  structured red-flag report

Run locally
-----------
    # default: Ollama backend (free, local)
    streamlit run review_app.py

    # switch to OpenAI for final testing
    export LLM_BACKEND=openai
    export OPENAI_API_KEY=sk-...
    streamlit run review_app.py

Ollama one-time setup
---------------------
    brew install ollama
    ollama pull llama3.2
    ollama serve
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from stage1_classical.inference import load_models, predict, preprocess
from stage2_llm.evidence_extractor import extract_evidence

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Job Detection — Two-Stage Review",
    page_icon="🕵️",
    layout="wide",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Posting Metadata")
    has_logo = st.checkbox("Has company logo", value=False)
    has_profile = st.checkbox("Has company profile", value=False)

    st.divider()

    st.subheader("LLM Backend")
    backend_choice = st.radio(
        "Stage 2 backend",
        options=["ollama (local)", "openai (API)"],
        index=0,
        help="Ollama is free and runs locally. Switch to OpenAI for production.",
    )
    backend = "ollama" if backend_choice.startswith("ollama") else "openai"

    if backend == "ollama":
        ollama_model = st.text_input("Ollama model", value="llama3.2")
        os.environ["OLLAMA_MODEL"] = ollama_model
        st.caption("Make sure `ollama serve` is running.")
    else:
        openai_key = st.text_input("OpenAI API key", type="password",
                                   value=os.getenv("OPENAI_API_KEY", ""))
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        openai_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
        os.environ["OPENAI_MODEL"] = openai_model
        st.caption("gpt-4o-mini is cheapest (~$0.00006 / posting).")

    st.divider()
    st.caption("Stage 1: Hybrid SVM (PR-AUC 0.9478, threshold 0.40)")

# ── main ──────────────────────────────────────────────────────────────────────
st.title("🕵️ Fraud Job Detection — Two-Stage Review")
st.caption(
    "**Stage 1** (SVM) flags postings fast. "
    "**Stage 2** (LLM) explains *why* for human review."
)

DEFAULT_TEXT = (
    "Seeking a remote administrative assistant. Earn weekly income from home. "
    "No experience required. Flexible hours, immediate start. "
    "Apply now and send your CV for same-day consideration."
)

job_text = st.text_area(
    "Paste a job posting",
    value=DEFAULT_TEXT,
    height=220,
    placeholder="Paste the full job description here…",
)

col_run1, col_run2, _ = st.columns([1, 1, 3])
run_stage1 = col_run1.button("▶ Run Stage 1", type="primary", use_container_width=True)
run_both   = col_run2.button("▶ Run Both Stages", type="secondary", use_container_width=True)

# ── load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def _load():
    return load_models()

lr_model, hybrid_model = _load()

# ── Stage 1 ───────────────────────────────────────────────────────────────────
if run_stage1 or run_both:
    if not job_text.strip():
        st.error("Please paste a non-empty job description.")
        st.stop()

    with st.spinner("Stage 1 — running classical classifier…"):
        result = predict(
            job_text=job_text,
            has_company_logo=has_logo,
            has_company_profile=has_profile,
            lr_model=lr_model,
            hybrid_model=hybrid_model,
        )

    # store in session state so Stage 2 can use it
    st.session_state["stage1_result"] = result
    st.session_state["job_text"] = job_text

    # ── Stage 1 output ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Stage 1 — Classical Classifier Output")

    risk_color = "#d62828" if result.risk_label == "High Risk" else "#2d6a4f"
    st.markdown(
        f"<h3 style='color:{risk_color};margin:0'>{result.risk_label}</h3>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Probability", f"{result.fraud_prob:.1%}")
    m2.metric("Company Logo", "✅" if has_logo else "❌")
    m3.metric("Company Profile", "✅" if has_profile else "❌")
    st.caption(f"Preprocessing: {result.preprocessing_note}")

    left, right = st.columns(2)

    with left:
        st.markdown("**Top Fraud-Leaning Terms** *(LR coefficients)*")
        if result.fraud_terms:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(result.fraud_terms).rename(
                    columns={"feature": "Term", "coefficient": "LR coef"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No strong fraud-leaning terms matched.")

        with st.expander("Legitimacy-leaning terms"):
            if result.legit_terms:
                st.dataframe(
                    pd.DataFrame(result.legit_terms).rename(
                        columns={"feature": "Term", "coefficient": "LR coef"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("None found.")

    with right:
        st.markdown("**Highlighted Trigger Terms**")
        if result.highlighted_html:
            st.markdown(
                f"<div style='line-height:1.9;font-size:0.9rem'>{result.highlighted_html}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.code(result.clean_text[:1500], language="text")

# ── Stage 2 ───────────────────────────────────────────────────────────────────
if run_both or st.button(
    "🤖 Generate LLM Red-Flag Report (Stage 2)",
    disabled="stage1_result" not in st.session_state,
    help="Run Stage 1 first, then generate the LLM evidence report.",
):
    if "stage1_result" not in st.session_state:
        st.warning("Run Stage 1 first.")
        st.stop()

    result = st.session_state["stage1_result"]
    raw_text = st.session_state.get("job_text", job_text)
    trigger_terms = [t["feature"] for t in result.fraud_terms]

    with st.spinner(f"Stage 2 — asking {backend} to analyse evidence…"):
        try:
            report = extract_evidence(
                job_text=raw_text,
                fraud_prob=result.fraud_prob,
                trigger_terms=trigger_terms,
                mode=backend,
            )
            st.session_state["stage2_report"] = report
        except Exception as e:
            st.error(f"Stage 2 failed: {e}")
            st.stop()

    # ── Stage 2 output ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Stage 2 — LLM Red-Flag Report")
    st.caption(f"Generated by: `{report.backend_used}`")

    # credibility score bar
    score = report.credibility_score
    score_color = "#d62828" if score <= 2 else ("#f4a261" if score == 3 else "#2d6a4f")
    st.markdown(
        f"**Credibility Score: "
        f"<span style='color:{score_color};font-size:1.2rem'>{score}/5</span>**"
        f" &nbsp; {'⚠️' * (6 - score) if score else ''}",
        unsafe_allow_html=True,
    )
    st.markdown(f"> {report.verdict_rationale}")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("🔴 **Language Signals**")
        if report.language_signals:
            for s in report.language_signals:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with col_b:
        st.markdown("🟡 **Structural Gaps**")
        if report.structural_gaps:
            for s in report.structural_gaps:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with col_c:
        st.markdown("🟠 **Urgency Flags**")
        if report.urgency_flags:
            for s in report.urgency_flags:
                st.markdown(f"- {s}")
        else:
            st.write("None identified.")

    with st.expander("Raw LLM response"):
        st.code(report.raw_response, language="json")

    # ── human review decision ─────────────────────────────────────────────────
    st.divider()
    st.subheader("👤 Human Review Decision")
    decision = st.radio(
        "Your decision on this posting:",
        ["Pending review", "Confirm Fraud — Remove", "Mark Legitimate — Keep"],
        horizontal=True,
    )
    notes = st.text_area("Review notes (optional)", height=80)
    if st.button("💾 Log Decision", use_container_width=False):
        import json, datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fraud_prob": result.fraud_prob,
            "risk_label": result.risk_label,
            "credibility_score": report.credibility_score,
            "decision": decision,
            "notes": notes,
            "backend": report.backend_used,
        }
        log_path = Path(__file__).parent / "data" / "review_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        st.success(f"Decision logged → `data/review_log.jsonl`")

    st.divider()
    st.caption(
        "Stage 1 (SVM hybrid, PR-AUC 0.9478) filters postings at scale. "
        "Stage 2 (LLM) provides explainable evidence for borderline or high-risk cases only, "
        "reducing API cost by ~80% vs running LLM on every posting."
    )
