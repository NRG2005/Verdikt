import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path
from datasketch import MinHash

log = logging.getLogger(__name__)

MINHASH_PERMS        = 128
SIMILARITY_THRESHOLD = 0.80
CASE_MEMORY_FILE     = Path(__file__).parent.parent / "data" / "case_memory.json"


def extract_features(tx: dict) -> set:
    """
    Updated for new 31-column schema.
    Uses beneficiary_id when available (more precise than receiver_account_external).
    Uses is_cross_border flag directly.
    """
    amount_inr  = float(tx.get('amount_inr', 0))
    amount_band = f"amt_{int(amount_inr // 1000)}k"

    # Use beneficiary_id if present (new field), else fall back to receiver_account_external
    receiver_id = (
        tx.get('beneficiary_id')
        or tx.get('receiver_account_id')
        or tx.get('receiver_account_external')
        or 'UNK'
    )
    # Normalise empty string to UNK
    if not receiver_id:
        receiver_id = 'UNK'

    features = {
        amount_band,
        f"channel_{tx.get('channel', 'UNK')}",
        f"purpose_{tx.get('purpose_code', 'UNK')}",
        f"rcv_{receiver_id}",
        f"sender_{tx.get('sender_account_id', 'UNK')}",
    }

    # Add cross-border flag as a feature — SWIFT/cross-border transactions
    # should never short-circuit against domestic ones
    if tx.get('is_cross_border') is True or tx.get('is_cross_border') == '1':
        features.add("cross_border_YES")

    return features


def make_minhash(features: set) -> MinHash:
    m = MinHash(num_perm=MINHASH_PERMS)
    for f in sorted(features):
        m.update(f.encode("utf-8"))
    return m


def load_case_memory() -> list:
    """Loads completed cases from local JSON file."""
    if not CASE_MEMORY_FILE.exists():
        return []
    with open(CASE_MEMORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_case_memory(cases: list) -> None:
    """Saves cases to local JSON file."""
    CASE_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CASE_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)


def query_case_memory(tx: dict) -> dict | None:
    """
    Searches case memory for the most similar past transaction.
    Returns the best match if similarity >= 0.80, else None.

    Short-circuit fires when:
      1. A match is found (score >= 0.80)
      2. AND the regulation hash hasn't changed since that verdict
    """
    cases = load_case_memory()
    if not cases:
        return None

    new_features = extract_features(tx)
    new_mh       = make_minhash(new_features)
    best, best_score = None, 0.0

    for case in cases:
        stored_features = set(case.get("feature_set", []))
        if not stored_features:
            continue
        stored_mh = make_minhash(stored_features)
        score     = new_mh.jaccard(stored_mh)
        if score > best_score:
            best_score, best = score, case

    if best_score >= SIMILARITY_THRESHOLD:
        best["_similarity_score"] = best_score
        log.info(
            f"Memory hit: {tx.get('tx_id')} matched "
            f"{best.get('tx_id')} (score={best_score:.3f})"
        )
        return best

    return None


def store_case(state: dict) -> None:
    """
    Saves a completed case to memory so future similar transactions
    can match against it, AND so the L5 review page can list/render it.
    Called after a case reaches final_status.
    """
    tx = state.get("tx_payload") or {}
    confidence = state.get("confidence")
    # Same single boundary api.py's L4/L5 routing uses: confidence >= 0.70
    # auto-generates a packet, below that goes to human review. Recomputed
    # here (not read from state) since there's no single stored
    # confidence_band field today. (Previously used a 0.50-0.90 band that
    # didn't match what api.py's live routing actually did -- see api.py's
    # L4/L5 section for that history.)
    needs_review = confidence is not None and confidence < 0.70

    cases = load_case_memory()
    cases.append({
        "tx_id":                   state["tx_id"],
        "case_id":                 state["case_id"],
        "feature_set":             build_feature_set(tx),
        "regulation_version_hash": state.get("regulation_hash_current"),
        "final_status":            state.get("final_status") or state.get("verdict"),
        "confidence":              confidence,
        "str_pdf_url":             state.get("str_pdf_url"),
        # Dossier fields for the L5 review page (view of the case, not used
        # by query_case_memory's MinHash matching, which only reads
        # feature_set above).
        "sender_name":             tx.get("sender_name", ""),
        "receiver_name":           tx.get("receiver_name", ""),
        "amount_inr":              tx.get("amount_inr"),
        "channel":                 tx.get("channel", ""),
        "delivery_status":         tx.get("delivery_status", ""),
        "dispute_reason":          tx.get("dispute_reason", ""),
        "triggers_fired":          state.get("triggers_fired") or [],
        "verdict":                 state.get("verdict"),
        "explanation":             state.get("explanation", ""),
        "citation_trail":          state.get("citation_trail"),
        "l4_disposition":          state.get("l4_disposition"),
        "l4_recommended_action":   state.get("l4_recommended_action"),
        "needs_review":            needs_review,
        "reviewer_decision":       state.get("reviewer_decision"),
        "reviewer_id":             state.get("reviewer_id"),
        "stored_at":               __import__("datetime").datetime.utcnow().isoformat() + "Z",
    })
    save_case_memory(cases)
    log.info(f"Stored case {state['tx_id']} in memory ({len(cases)} total)")


def build_feature_set(tx: dict) -> list:
    """Returns sorted feature list for storing in case memory."""
    return sorted(extract_features(tx))