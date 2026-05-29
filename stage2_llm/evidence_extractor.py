"""
Stage 2 — LLM Evidence Extractor
==================================
Generates a structured red-flag summary for job postings flagged by Stage 1.

Backend switching (Ollama ↔ OpenAI)
-------------------------------------
Both backends use the same openai Python package.
Switch by changing the ENV variable LLM_BACKEND:

    export LLM_BACKEND=ollama    # default — local, free
    export LLM_BACKEND=openai    # production — needs OPENAI_API_KEY

Or pass mode= explicitly to build_client() / extract_evidence().

Ollama setup (one-time)
------------------------
    brew install ollama
    ollama pull llama3.2        # or mistral, phi3, etc.
    ollama serve                # starts on http://localhost:11434

Cost note
----------
Using gpt-4o-mini on OpenAI costs ~$0.00015 / 1k input tokens.
A typical job posting is ~400 tokens → ~$0.00006 per call.
Only call Stage 2 for postings where fraud_prob >= threshold (Stage 1 filters first).
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Literal

# openai package works for both Ollama and OpenAI
from openai import OpenAI

# ── types ─────────────────────────────────────────────────────────────────────
Backend = Literal["ollama", "openai"]

DEFAULT_OLLAMA_MODEL = "llama3.2"       # change to mistral, phi3, etc.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"   # cheapest capable model


@dataclass
class EvidenceReport:
    language_signals: list[str]       # suspicious phrasing patterns
    structural_gaps: list[str]        # missing credibility elements
    urgency_flags: list[str]          # pressure / urgency tactics
    credibility_score: int            # 1 (very suspicious) – 5 (credible)
    verdict_rationale: str            # one-sentence overall summary
    raw_response: str                 # raw LLM output for debugging
    backend_used: str                 # "ollama/llama3.2" or "openai/gpt-4o-mini"


# ── client factory ────────────────────────────────────────────────────────────
def build_client(mode: Backend | None = None) -> tuple[OpenAI, str]:
    """
    Returns (client, model_name).

    Priority: explicit mode arg → LLM_BACKEND env var → "ollama" default.
    """
    if mode is None:
        mode = os.getenv("LLM_BACKEND", "ollama")  # type: ignore[assignment]

    if mode == "ollama":
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",   # Ollama ignores this but the SDK requires a value
        )
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        return client, model

    elif mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Run: export OPENAI_API_KEY=sk-..."
            )
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return client, model

    else:
        raise ValueError(f"Unknown backend: {mode!r}. Use 'ollama' or 'openai'.")


# ── prompt ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = textwrap.dedent("""
    You are a fraud analyst reviewing job postings for a compliance team.
    Your job is to identify red flags that suggest a posting may be fraudulent.

    Respond ONLY with valid JSON matching this exact schema — no extra text:
    {
      "language_signals":   [<list of suspicious phrasing patterns, max 4 items>],
      "structural_gaps":    [<list of missing credibility elements, max 4 items>],
      "urgency_flags":      [<list of pressure or urgency tactics, max 3 items>],
      "credibility_score":  <integer 1–5, where 1=very suspicious, 5=credible>,
      "verdict_rationale":  "<one sentence summary>"
    }

    Be specific. Quote or paraphrase actual phrases from the posting.
    If a category has no findings, return an empty list [].
""").strip()


def _build_user_prompt(
    job_text: str,
    fraud_prob: float,
    trigger_terms: list[str],
) -> str:
    terms_str = ", ".join(trigger_terms[:12]) if trigger_terms else "none identified"
    text_snippet = job_text[:2000].strip()
    if len(job_text) > 2000:
        text_snippet += "\n[... truncated ...]"

    return textwrap.dedent(f"""
        CLASSIFIER OUTPUT
        -----------------
        Fraud probability: {fraud_prob:.1%}
        Top trigger terms from classical model: {terms_str}

        JOB POSTING TEXT
        ----------------
        {text_snippet}

        Now produce the JSON red-flag report.
    """).strip()


# ── main extract function ─────────────────────────────────────────────────────
def extract_evidence(
    job_text: str,
    fraud_prob: float,
    trigger_terms: list[str],
    mode: Backend | None = None,
    timeout: float = 60.0,
) -> EvidenceReport:
    """
    Call the LLM and return a structured EvidenceReport.

    Parameters
    ----------
    job_text : str
        Raw or cleaned job posting text.
    fraud_prob : float
        Fraud probability from Stage 1 (0–1).
    trigger_terms : list[str]
        Top fraud-leaning terms from LR model (for context injection).
    mode : "ollama" | "openai" | None
        Backend to use. None → reads LLM_BACKEND env var → defaults to "ollama".
    timeout : float
        Request timeout in seconds. Increase if Ollama is slow on first load.

    Returns
    -------
    EvidenceReport
    """
    client, model = build_client(mode)
    user_prompt = _build_user_prompt(job_text, fraud_prob, trigger_terms)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,    # low temp → consistent structured output
        max_tokens=512,
        timeout=timeout,
    )

    raw = response.choices[0].message.content or ""

    # ── parse JSON ────────────────────────────────────────────────────────────
    try:
        # strip markdown code fences if the model wraps in ```json ... ```
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # graceful fallback — wrap raw text so the app doesn't crash
        data = {
            "language_signals": [],
            "structural_gaps": [],
            "urgency_flags": [],
            "credibility_score": 0,
            "verdict_rationale": raw[:300],
        }

    return EvidenceReport(
        language_signals=data.get("language_signals", []),
        structural_gaps=data.get("structural_gaps", []),
        urgency_flags=data.get("urgency_flags", []),
        credibility_score=int(data.get("credibility_score", 0)),
        verdict_rationale=data.get("verdict_rationale", ""),
        raw_response=raw,
        backend_used=f"{mode or os.getenv('LLM_BACKEND', 'ollama')}/{model}",
    )


# ── quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = (
        "We are hiring remote data entry clerks. Earn $500-$1000 per week "
        "working from home. No experience required. Immediate start. "
        "Send your personal details and a small registration fee to get started."
    )
    print("Testing Stage 2 evidence extractor...")
    print(f"Backend: {os.getenv('LLM_BACKEND', 'ollama')}\n")

    report = extract_evidence(
        job_text=sample_text,
        fraud_prob=0.92,
        trigger_terms=["earn", "no experience required", "registration fee", "immediate"],
    )
    print(f"Credibility score : {report.credibility_score}/5")
    print(f"Verdict           : {report.verdict_rationale}")
    print(f"Language signals  : {report.language_signals}")
    print(f"Structural gaps   : {report.structural_gaps}")
    print(f"Urgency flags     : {report.urgency_flags}")
    print(f"Backend used      : {report.backend_used}")
