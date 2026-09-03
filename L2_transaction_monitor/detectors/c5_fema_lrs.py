"""
C5 — Delivery-Confirmation / Dispute-Evidence Mismatch  (Track 02 retrofit)

ORIGINAL ROLE (AML pipeline): FEMA/LRS cross-border remittance-ceiling
monitoring. That signal has no fraud/chargeback analog (there is no merchant-
side equivalent of a statutory annual outward-remittance cap), so this
category's core logic has been replaced rather than relabeled — see the
retrofit plan for the full rationale.

NEW ROLE: once a transaction has an actual dispute filed against it
(`dispute_filed == "1"`), this detector reads the merchant's own delivery/
fulfilment record and reports which way the evidence points:

  - delivery_status == DELIVERED     -> strong evidence the goods/service
                                         reached the customer despite the
                                         dispute (classic friendly-fraud
                                         signature) -> favors CONTEST
  - delivery_status == IN_TRANSIT    -> confirmation is incomplete -> the
                                         evidence is ambiguous, still worth
                                         routing to human/L3 review
  - NOT_DELIVERED / RETURNED / n/a   -> no delivery evidence exists -> favors
                                         CONCEDE (contesting would be baseless)

This fires on EVERY disputed transaction (all three branches), not just the
contestable ones — every filed dispute needs a routed decision, and this is
the signal that tells L3/L4 which decision the evidence supports. It does
NOT use any historical/account-level dispute-count field, deliberately: those
would only be knowable after other disputes on the same account are already
labeled, which risks leaking the very outcome being predicted. Everything
here is a fact about the CURRENT transaction and its OWN fulfilment record.
"""

TRIGGER_FAVORS_CONTEST = "C5_evidence_favors_contest"
TRIGGER_AMBIGUOUS = "C5_evidence_ambiguous"
TRIGGER_FAVORS_CONCEDE = "C5_evidence_favors_concede"

SCORE_FAVORS_CONTEST = 0.75
SCORE_AMBIGUOUS = 0.45
SCORE_FAVORS_CONCEDE = 0.30


def evaluate_row(row, dl):
    """Return {fired, score, trigger} for one unified transactions.csv row.

    `dl` (DataLayer) is accepted for interface parity with the other five
    detectors but isn't needed here — every input this detector reads
    (`dispute_filed`, `delivery_status`) lives on the row itself.
    """
    if str(row.get("dispute_filed", "0")) != "1":
        return {"fired": False, "score": 0.0, "trigger": None}

    delivery = (row.get("delivery_status") or "").strip().upper()

    if delivery == "DELIVERED":
        return {"fired": True, "score": SCORE_FAVORS_CONTEST, "trigger": TRIGGER_FAVORS_CONTEST}
    if delivery == "IN_TRANSIT":
        return {"fired": True, "score": SCORE_AMBIGUOUS, "trigger": TRIGGER_AMBIGUOUS}
    # NOT_DELIVERED, RETURNED, NOT_APPLICABLE, blank, or anything unexpected
    return {"fired": True, "score": SCORE_FAVORS_CONCEDE, "trigger": TRIGGER_FAVORS_CONCEDE}


def recommended_action(trigger):
    """Map a C5 trigger to the recommended dispute-response action — used by
    L4's report generator to pre-fill its recommended-action field."""
    return {
        TRIGGER_FAVORS_CONTEST: "CONTEST_WITH_EVIDENCE",
        TRIGGER_AMBIGUOUS: "ESCALATE_TO_REVIEW",
        TRIGGER_FAVORS_CONCEDE: "CONCEDE",
    }.get(trigger, "ESCALATE_TO_REVIEW")
