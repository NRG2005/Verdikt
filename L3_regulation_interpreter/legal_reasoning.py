"""
L3: GPT-5.1 Legal Reasoning

Computes the 4 sub-scores and generates a citation trail.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from .llm_client import chat_json, is_llm_configured
from .hybrid_retrieval import _compute_retrieval_match


SYSTEM_PROMPT = """
You are the dispute-evidence reasoning engine for a merchant chargeback response pipeline.

Your job is to reason over:
- transaction and dispute facts (including the delivery-confirmation status and
  any risk signals raised upstream)
- retrieved card-network dispute-category and compelling-evidence guidance
- known dispute typologies (fraud, friendly fraud, merchant error)

You must return ONLY valid JSON.
Be conservative, cite the retrieved material, and never invent a rule that is
not present in the retrieved chunks.

CRITICAL INSTRUCTION FOR L2 TRIGGERS:
If the transaction has an 'l2_triggers_fired' flag (e.g., C5_evidence_favors_contest,
C5_evidence_favors_concede), you MUST treat the upstream signal as already-verified
evidence, not as something to re-derive from scratch. For example, if C5 reports
delivery-confirmed evidence favoring a contest, and a retrieved rule states that
confirmed delivery is compelling evidence for a consumer-dispute-category case,
score the case highly (e.g., 0.80+) for a CONTEST verdict. If C2/C6 report
account-takeover-style signals (known fraud IP, impossible travel, new-device
takeover), treat that as evidence supporting FRAUD-category concession, not
contest, per the corpus guidance that delivery evidence does not rebut a
fraud-category claim.

CRITICAL INSTRUCTION ON JARGON:
When writing the "explanation", DO NOT use internal pipeline jargon or abbreviations such as C1, C2, C5, L1, L2, L3, or L4. The final reader of the report (a dispute-ops analyst or the merchant) will not understand these terms. Instead of saying "L2 trigger C5_evidence_favors_contest fired", write "Delivery was confirmed to the cardholder's address despite the dispute" or describe the actual behavior. Use clear, professional, plain language to explain the reasoning and the patterns observed.
""".strip()


USE_CASES = [
    {
        "name": "friendly_fraud",
        "description": "The cardholder disputes a transaction they genuinely made and received the benefit of (goods delivered, service used), typically to avoid payment rather than because the charge was actually unauthorized.",
    },
    {
        "name": "true_fraud",
        "description": "The payment credential was genuinely compromised (stolen card, account takeover); the dispute is valid and no legitimate delivery/fulfilment evidence can rebut it.",
    },
    {
        "name": "merchant_error",
        "description": "A genuine merchant-side or processing fault: item never shipped, duplicate charge, or an incorrect amount charged.",
    },
    {
        "name": "delivery_dispute",
        "description": "The cardholder disputes non-receipt or a fulfilment defect where delivery status is ambiguous (in-transit, unconfirmed, or delivered to an unverified address) rather than clearly proven either way.",
    },
]


def _serialize_chunks(regulation_chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Limit to top 2 chunks and truncate content to save tokens
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "document_id": chunk.get("document_id"),
            "title": chunk.get("title"),
            "section_id": chunk.get("section_id"),
            "content": str(chunk.get("content", ""))[:3000],  # Increased to prevent cutting off rules
            "retrieval_score": chunk.get("retrieval_score"),
        }
        for chunk in regulation_chunks[:3]
    ]


def _build_reasoning_prompt(event: Dict[str, Any], regulation_chunks: Sequence[Dict[str, Any]]) -> str:
    return f"""
Analyze this transaction case against the provided regulations.

Transaction:
{json.dumps(event)}

Suspicious Typologies:
{json.dumps(USE_CASES)}

Retrieved Regulations (Top Matches):
{json.dumps(_serialize_chunks(regulation_chunks))}

Return ONLY JSON with this exact structure:
{{
  "retrieval_match": <0 to 1, how well retrieved chunks match case facts>,
  "rule_applicability": <0 to 1, how clearly retrieved rules apply>,
  "evidence_sufficiency": <0 to 1, transaction facts support defensible decision>,
  "precedent_confidence": <0 to 1, pattern resembles suspicious typologies>,
  "applicable_use_cases": ["..."],
  "applicable_rules": [
    {{
      "document_id": "...",
      "section_id": "...",
      "reason": "..."
    }}
  ],
  "citation_trail": [
    {{
      "chunk_id": "...",
      "excerpt": "short excerpt starting with the rule designation (e.g., Article 22.1) followed by the rule text",
      "why_it_matters": "..."
    }}
  ],
  "verdict": "clear" | "review" | "suspicious",
  "explanation": "...",
  "final_score": <0 to 1>
}}

Scoring guidance:
Final score MUST reflect the weighted legal confidence after reasoning.
If the rules are weakly related or evidence is thin, lower the score.

CRITICAL CALIBRATION INSTRUCTION FOR rule_applicability:
Do not score rule_applicability based on whether the retrieved text is
topically adjacent (e.g. "this is also about UPI disputes") -- score it on
whether the retrieved rule specifically and directly decides THIS fact
pattern. Use this concrete scale:
  0.8-1.0: the retrieved rule's own text directly names the specific
    scenario in front of you (e.g. the exact liability trigger, the exact
    evidence requirement) and tells you what outcome follows.
  0.4-0.7: the retrieved rule is genuinely relevant background (e.g. it
    establishes the general dispute-lifecycle stages or a related timing
    rule) but does not itself resolve whether THIS transaction should be
    contested or conceded -- you are still bridging a real inferential gap.
  0.0-0.3: the retrieved rule is only topically adjacent (mentions UPI
    disputes, chargebacks, or evidence in general) without addressing the
    specific fact pattern (e.g. a procedural TAT/format rule when the real
    question is fraud-vs-friendly-fraud liability, or vice versa).
A retrieval hit is not the same as a rule applying. If you find yourself
reasoning "well, it's at least about disputes in general" -- that is a 0.2-0.4
case, not a 0.7+ case. Most real cases in this domain sit in the 0.4-0.7
band, because a single short circular rarely fully decides a fact pattern by
itself; treat anything above 0.7 as reserved for genuinely direct, on-point
matches, not the default.
""".strip()

def _enrich_citation_trail(citation_trail, all_chunks):
    """Add rule designation to each citation entry.
    citation_trail: list of dicts with at least "chunk_id" and "excerpt".
    all_chunks: list of chunk dicts (azure + nomic) containing metadata.
    Returns a new list with an added "rule_designation" field and ensures the excerpt starts with it.
    """
    # Build a mapping from chunk_id to chunk metadata for quick lookup
    chunk_map = {c.get("chunk_id"): c for c in all_chunks}
    enriched = []
    for entry in citation_trail:
        chunk_id = entry.get("chunk_id")
        chunk = chunk_map.get(chunk_id, {})
        # Prefer section_id, fallback to document_id, then title
        designation = chunk.get("section_id") or chunk.get("document_id") or chunk.get("title") or ""
        # Clean designation (strip whitespace)
        designation = str(designation).strip()
        # Ensure excerpt starts with designation
        excerpt = entry.get("excerpt", "")
        if designation and not excerpt.startswith(designation):
            excerpt = f"{designation} – {excerpt}" if designation else excerpt
        enriched_entry = {
            **entry,
            "rule_designation": designation,
            "excerpt": excerpt,
        }
        enriched.append(enriched_entry)
    return enriched


def _clamp01(v, default=0.0):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _finalize_score(analysis, chunks):
    """
    Replaces two things the raw LLM output can't be trusted for, with real
    computed/deterministic values:

    1. `retrieval_match` -- the prompt asks the LLM to estimate this itself
       from the chunks it's shown, but we already COMPUTE this precisely
       (cosine similarity locally, RRF fusion via Azure) before the LLM ever
       sees anything. Observed in practice: real computed retrieval_match
       0.929 (a genuinely strong match) vs. the LLM's own self-reported
       guess of 0.2 for the same case -- a wrong, redundant re-estimate that
       then silently drags the final score down with it.
    2. `final_score` -- previously the LLM's own free-floating number, with
       no formula tying it to the 4 sub-scores at all (the prompt only said
       "reflect the weighted confidence", not an actual weight). That's why
       the same case could score 0.55 on one call and 0.75 on the next --
       it wasn't measuring anything stable, just "however conservative the
       model felt that call." Now final_score is a real, fixed formula:
       35% real retrieval_match + 30% rule_applicability + 20% evidence_sufficiency
       + 15% precedent_confidence (the LLM still judges these three -- that's
       genuine legal reasoning, not something we can compute directly).
    """
    if not analysis:
        return analysis
    real_retrieval_match = _compute_retrieval_match(chunks)
    rule_applicability = _clamp01(analysis.get("rule_applicability"))
    evidence_sufficiency = _clamp01(analysis.get("evidence_sufficiency"))
    precedent_confidence = _clamp01(analysis.get("precedent_confidence"))
    computed_score = round(
        0.35 * real_retrieval_match
        + 0.30 * rule_applicability
        + 0.20 * evidence_sufficiency
        + 0.15 * precedent_confidence,
        4,
    )
    analysis["llm_self_reported_retrieval_match"] = analysis.get("retrieval_match")
    analysis["llm_self_reported_final_score"] = analysis.get("final_score")
    analysis["retrieval_match"] = round(real_retrieval_match, 4)
    analysis["final_score"] = computed_score
    return analysis


def generate_legal_analysis(event, retrieval):
    """
    Uses a large language model to analyze the transaction against the retrieved regulations.
    Performs Dual-Evaluation: runs LLM independently on Azure AI Search chunks and Local Nomic chunks,
    and returns the verdict with the highest confidence score.
    """
    print("L3: Applying dual legal reasoning (Azure AI vs Local Nomic)...")

    azure_chunks = retrieval.get("chunks", [])
    nomic_chunks = retrieval.get("nomic_chunks", [])

    try:
        # The existing try/except blocks for each backend remain unchanged.
        azure_analysis = None
        nomic_analysis = None
        fallback_reason = "LLM not configured."

        # chat_json() already falls back to local Ollama when GEMINI_API_KEY
        # is unset, so always attempt the call rather than gating on
        # is_llm_configured() (which only checks for the Gemini key).
        try:
            if azure_chunks:
                azure_analysis = chat_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=_build_reasoning_prompt(event, azure_chunks),
                )
                azure_analysis["backend_used"] = "azure_ai_search"
                azure_analysis = _finalize_score(azure_analysis, azure_chunks)
        except Exception as exc:
            print(f"L3: Azure evaluation failed: {exc}")

        try:
            if nomic_chunks:
                nomic_analysis = chat_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=_build_reasoning_prompt(event, nomic_chunks),
                )
                nomic_analysis["backend_used"] = "local_nomic_search"
                nomic_analysis = _finalize_score(nomic_analysis, nomic_chunks)
        except Exception as exc:
            print(f"L3: Local Nomic evaluation failed: {exc}")

        # Determine the winner (highest final_score)
        winner = None
        if azure_analysis and nomic_analysis:
            if float(nomic_analysis.get("final_score", 0)) > float(azure_analysis.get("final_score", 0)):
                print(f"L3 Dual-Eval: Local Nomic model scored higher confidence ({nomic_analysis.get('final_score')} vs {azure_analysis.get('final_score')}). Using Local!")
                winner = nomic_analysis
            else:
                print("L3 Dual-Eval: Azure model scored higher or equal confidence. Using Azure!")
                winner = azure_analysis
        elif azure_analysis:
            winner = azure_analysis
        elif nomic_analysis:
            winner = nomic_analysis

        if winner:
            # Enrich citation_trail if present as a list
            if isinstance(winner.get("citation_trail"), list):
                all_chunks = azure_chunks + nomic_chunks
                winner["citation_trail"] = _enrich_citation_trail(winner.get("citation_trail", []), all_chunks)
            return winner

        # No successful analysis – fallback
        print("L3: Fallback triggered – no LLM analysis succeeded.")
        return {
            "retrieval_match": 0.0,
            "rule_applicability": 0.0,
            "evidence_sufficiency": 0.0,
            "precedent_confidence": 0.0,
            "citation_trail": f"Fallback: {fallback_reason}",
            "final_score": 0.0,
            "verdict": "review",
        }
    except Exception as outer_exc:
        # Catch any unexpected error and return a safe default
        print(f"L3: Unexpected error during legal analysis: {outer_exc}")
        return {
            "retrieval_match": 0.0,
            "rule_applicability": 0.0,
            "evidence_sufficiency": 0.0,
            "precedent_confidence": 0.0,
            "citation_trail": f"Unexpected error: {outer_exc}",
            "final_score": 0.0,
            "verdict": "error",
        }
