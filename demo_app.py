from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LR_MODEL_PATH = DATA_DIR / "lr_best_model.pkl"
HYBRID_MODEL_PATH = DATA_DIR / "svm_hybrid_best_model.pkl"


def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?:#)?url_[a-z0-9]{15,}(?:#)?", " ", text)
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r"\\S+@\\S+", " ", text)
    text = re.sub(r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b", " ", text)
    text = re.sub(r"[^\\w\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        return None


def preprocess_text(raw_text: str) -> tuple[str, str]:
    regex_cleaned = basic_clean(raw_text)
    nlp = load_spacy()
    if nlp is None:
        return regex_cleaned, "spaCy unavailable: using regex-cleaned text only."

    doc = nlp(regex_cleaned[:100000])
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    clean_text = " ".join(tokens).strip()
    if not clean_text:
        clean_text = regex_cleaned
        return clean_text, "spaCy returned empty text: fell back to regex-cleaned text."
    return clean_text, "Using regex cleaning + lemmatization + stopword removal."


@st.cache_resource(show_spinner=False)
def load_models():
    lr_model = joblib.load(LR_MODEL_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    return lr_model, hybrid_model


def matched_lr_features(clean_text: str, lr_model, top_n: int = 10) -> tuple[list[dict], list[dict]]:
    vectorizer = lr_model.named_steps["vectorizer"]
    classifier = lr_model.named_steps["classifier"]
    vocab = vectorizer.vocabulary_
    coefficients = classifier.coef_[0]
    tokens = clean_text.split()

    seen = set()
    matches = []

    for token in tokens:
        if token in vocab and token not in seen:
            idx = vocab[token]
            matches.append({"feature": token, "coefficient": float(coefficients[idx])})
            seen.add(token)

    for left, right in zip(tokens, tokens[1:]):
        bigram = f"{left} {right}"
        if bigram in vocab and bigram not in seen:
            idx = vocab[bigram]
            matches.append({"feature": bigram, "coefficient": float(coefficients[idx])})
            seen.add(bigram)

    matches = sorted(matches, key=lambda item: item["coefficient"], reverse=True)
    fraud_terms = [m for m in matches if m["coefficient"] > 0][:top_n]
    legit_terms = sorted(
        [m for m in matches if m["coefficient"] < 0],
        key=lambda item: item["coefficient"],
    )[:top_n]
    return fraud_terms, legit_terms


def highlight_terms(clean_text: str, terms: Iterable[str]) -> str:
    highlighted = clean_text
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"\\b{re.escape(term)}\\b", re.IGNORECASE)
        highlighted = pattern.sub(
            lambda match: f"<mark style='background-color:#ffd166;padding:0 0.1rem;'>{match.group(0)}</mark>",
            highlighted,
        )
    return highlighted


st.set_page_config(
    page_title="Fake Job Detection Demo",
    page_icon="🕵️",
    layout="wide",
)

st.title("Interactive Fake Job Detection Demo")
st.caption("Final model: Hybrid SVM (TF-IDF + 2 credibility indicators)")

with st.sidebar:
    st.subheader("Posting Context")
    has_company_logo = st.checkbox("Posting includes a company logo", value=False)
    has_company_profile = st.checkbox("Posting includes a company profile", value=False)
    st.caption(
        "These two metadata fields are part of the final hybrid model. "
        "If you only have raw text, leave them unchecked."
    )

default_text = (
    "Seeking a remote administrative assistant. Earn weekly income from home. "
    "No experience required. Apply now with your resume for immediate consideration."
)
job_text = st.text_area(
    "Paste a job description",
    value=default_text,
    height=240,
)

run = st.button("Run Fraud Detection", type="primary", use_container_width=True)

if run:
    if not job_text.strip():
        st.error("Paste a non-empty job description first.")
    else:
        lr_model, hybrid_model = load_models()
        clean_text, preprocessing_note = preprocess_text(job_text)

        hybrid_input = pd.DataFrame(
            [
                {
                    "clean_text": clean_text,
                    "has_company_logo": int(has_company_logo),
                    "has_company_profile": int(has_company_profile),
                }
            ]
        )
        fraud_probability = float(hybrid_model.predict_proba(hybrid_input)[0, 1])
        risk_label = "High Risk" if fraud_probability >= 0.40 else "Likely Legitimate"
        fraud_terms, legit_terms = matched_lr_features(clean_text, lr_model)

        left, right = st.columns([1, 1])
        with left:
            st.metric("Fraud Probability", f"{fraud_probability:.1%}")
            st.metric("Decision Label", risk_label)
            st.caption(preprocessing_note)

            st.subheader("Matched Fraud-Leaning Terms")
            if fraud_terms:
                st.dataframe(
                    pd.DataFrame(fraud_terms).rename(
                        columns={"feature": "Matched term", "coefficient": "LR coefficient"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No strong fraud-leaning LR terms matched this text.")

            with st.expander("Matched Legitimacy-Leaning Terms"):
                if legit_terms:
                    st.dataframe(
                        pd.DataFrame(legit_terms).rename(
                            columns={"feature": "Matched term", "coefficient": "LR coefficient"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("No legitimacy-leaning LR terms matched this text.")

        with right:
            st.subheader("Preprocessed Text")
            st.code(clean_text[:2500] or "(empty after preprocessing)", language="text")

            st.subheader("Highlighted Trigger Terms")
            if fraud_terms:
                highlighted = highlight_terms(clean_text, [item["feature"] for item in fraud_terms])
                st.markdown(
                    f"<div style='line-height:1.8'>{highlighted}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("No highlightable fraud-trigger terms were found in the cleaned text.")

        st.divider()
        st.caption(
            "Probability comes from the final hybrid SVM. "
            "Term explanations come from the Logistic Regression baseline for interpretability."
        )
