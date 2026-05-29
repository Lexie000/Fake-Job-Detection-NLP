"""
Stage 1 — Classical NLP Inference
==================================
Wraps the Hybrid SVM (TF-IDF + credibility metadata) and the LR baseline
into a single, importable interface.  demo_app.py and review_app.py both
import from here so all model logic lives in one place.

Usage
-----
from stage1_classical.inference import load_models, predict

lr_model, hybrid_model = load_models()
result = predict(
    job_text="Earn money from home, no experience needed...",
    has_company_logo=False,
    has_company_profile=False,
    lr_model=lr_model,
    hybrid_model=hybrid_model,
)
print(result["fraud_prob"], result["risk_label"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LR_MODEL_PATH = DATA_DIR / "lr_best_model.pkl"
HYBRID_MODEL_PATH = DATA_DIR / "svm_hybrid_best_model.pkl"

FRAUD_THRESHOLD = 0.40  # tuned on validation set


# ── data class for prediction output ─────────────────────────────────────────
@dataclass
class PredictionResult:
    fraud_prob: float
    risk_label: str          # "High Risk" | "Likely Legitimate"
    fraud_terms: list[dict]  # [{"feature": str, "coefficient": float}, ...]
    legit_terms: list[dict]
    clean_text: str
    preprocessing_note: str
    highlighted_html: str = field(default="")


# ── text preprocessing ────────────────────────────────────────────────────────
def _basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?:#)?url_[a-z0-9]{15,}(?:#)?", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(raw_text: str) -> tuple[str, str]:
    """
    Returns (clean_text, note).
    Tries spaCy lemmatization; falls back to regex-only if unavailable.
    """
    regex_cleaned = _basic_clean(raw_text)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        doc = nlp(regex_cleaned[:100_000])
        tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space
        ]
        clean = " ".join(tokens).strip() or regex_cleaned
        note = "regex cleaning + spaCy lemmatization + stopword removal"
    except Exception:
        clean = regex_cleaned
        note = "spaCy unavailable — regex cleaning only"
    return clean, note


# ── model loading ─────────────────────────────────────────────────────────────
def load_models(
    lr_path: Path = LR_MODEL_PATH,
    hybrid_path: Path = HYBRID_MODEL_PATH,
) -> tuple:
    """Load and return (lr_model, hybrid_model). Call once; cache the result."""
    lr_model = joblib.load(lr_path)
    hybrid_model = joblib.load(hybrid_path)
    return lr_model, hybrid_model


# ── LR interpretability ───────────────────────────────────────────────────────
def _extract_lr_terms(
    clean_text: str,
    lr_model,
    top_n: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Return (fraud_terms, legit_terms) sorted by |coefficient|."""
    vectorizer = lr_model.named_steps["vectorizer"]
    classifier = lr_model.named_steps["classifier"]
    vocab = vectorizer.vocabulary_
    coef = classifier.coef_[0]

    tokens = clean_text.split()
    seen: set[str] = set()
    matches: list[dict] = []

    for tok in tokens:
        if tok in vocab and tok not in seen:
            matches.append({"feature": tok, "coefficient": float(coef[vocab[tok]])})
            seen.add(tok)

    for a, b in zip(tokens, tokens[1:]):
        bigram = f"{a} {b}"
        if bigram in vocab and bigram not in seen:
            matches.append({"feature": bigram, "coefficient": float(coef[vocab[bigram]])})
            seen.add(bigram)

    fraud_terms = sorted([m for m in matches if m["coefficient"] > 0],
                         key=lambda x: x["coefficient"], reverse=True)[:top_n]
    legit_terms = sorted([m for m in matches if m["coefficient"] < 0],
                         key=lambda x: x["coefficient"])[:top_n]
    return fraud_terms, legit_terms


# ── term highlighting ─────────────────────────────────────────────────────────
def _highlight(clean_text: str, terms: list[str]) -> str:
    html = clean_text
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        html = pattern.sub(
            lambda m: (
                f"<mark style='background-color:#ffd166;"
                f"padding:0 0.1rem;border-radius:2px'>{m.group(0)}</mark>"
            ),
            html,
        )
    return html


# ── main predict function ─────────────────────────────────────────────────────
def predict(
    job_text: str,
    has_company_logo: bool,
    has_company_profile: bool,
    lr_model,
    hybrid_model,
    top_n: int = 10,
    clean_text: Optional[str] = None,
    preprocessing_note: Optional[str] = None,
) -> PredictionResult:
    """
    Run Stage 1 inference.

    Parameters
    ----------
    job_text : str
        Raw job posting text (title + description + etc.).
    has_company_logo : bool
        Credibility metadata indicator.
    has_company_profile : bool
        Credibility metadata indicator.
    lr_model, hybrid_model : sklearn Pipeline objects from load_models().
    top_n : int
        Number of top fraud / legit LR terms to return.
    clean_text : str, optional
        Pre-processed text. If None, preprocess() is called internally.
    preprocessing_note : str, optional
        Explanation string to pass through. Auto-filled if clean_text is None.

    Returns
    -------
    PredictionResult
    """
    if clean_text is None:
        clean_text, preprocessing_note = preprocess(job_text)

    # Hybrid SVM prediction
    hybrid_input = pd.DataFrame([{
        "clean_text": clean_text,
        "has_company_logo": int(has_company_logo),
        "has_company_profile": int(has_company_profile),
    }])
    fraud_prob = float(hybrid_model.predict_proba(hybrid_input)[0, 1])
    risk_label = "High Risk" if fraud_prob >= FRAUD_THRESHOLD else "Likely Legitimate"

    # LR-based interpretability
    fraud_terms, legit_terms = _extract_lr_terms(clean_text, lr_model, top_n)

    # Highlighted HTML
    highlighted_html = _highlight(
        clean_text, [t["feature"] for t in fraud_terms]
    )

    return PredictionResult(
        fraud_prob=fraud_prob,
        risk_label=risk_label,
        fraud_terms=fraud_terms,
        legit_terms=legit_terms,
        clean_text=clean_text,
        preprocessing_note=preprocessing_note or "",
        highlighted_html=highlighted_html,
    )
