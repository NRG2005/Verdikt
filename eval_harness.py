#!/usr/bin/env python3
"""
eval_harness.py — Track 02 precision/recall + false-positive-cost report.

Reports on the HELD-OUT split only (data/ground_truth.csv, split=="holdout").
`ground_truth.csv` is read here and ONLY here — no detector, no DataLayer, no
LLM prompt in this codebase reads it; it is not a pipeline input.

Two separate tasks are scored, because they are genuinely different jobs and
lumping them into one number would overstate the system:

  TASK A — Proactive risk detection (pre-dispute)
    Detectors: C1, C2, C3, C4, C6 (transaction-time anomaly signals).
    Target: is this transaction itself fraudulent (true_fraud), catchable
    BEFORE any dispute is ever filed. friendly_fraud is deliberately
    excluded from this target — by construction it has no transaction-time
    anomaly (the purchase legitimately happened), so it is undetectable at
    this stage. That is reported explicitly below, not hidden.

  TASK B — Dispute-evidence classification (post-dispute)
    Detector: C5 (delivery-confirmation signal), scored only on the subset
    of holdout transactions where a dispute was actually filed.
    Target: does the evidence direction C5 reports (CONTEST / CONCEDE /
    ESCALATE) match the correct_action ground truth.

False-positive cost: for Task A, a false positive means a legitimate
transaction gets pulled into manual review — cost = analyst review time.
A false negative means real fraud goes unflagged — cost = the fraud amount
itself. Both are reported using the transaction's own amount_inr, with the
review-cost-per-case assumption stated plainly (this is an assumption, not a
measurement — flagged as such in the output).
"""

import csv
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from L2_transaction_monitor.data_layer import DataLayer  # noqa: E402
from L2_transaction_monitor.orchestrator import monitor  # noqa: E402
from L2_transaction_monitor.detectors.c5_fema_lrs import recommended_action  # noqa: E402

import asyncio

REVIEW_COST_INR = 150.0  # ASSUMED analyst-minutes cost per manually reviewed case


def load_ground_truth(path="data/ground_truth.csv"):
    gt = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            gt[r["tx_id"]] = r
    return gt


async def run_pipeline(dl):
    results = {}
    for row in dl.transactions:
        results[row["tx_id"]] = await monitor(row, dl)
    return results


def task_a(dl, gt, results, holdout_ids):
    """Proactive risk detection: C1,C2,C3,C4,C6 only (C5 excluded — it only
    fires post-dispute and would trivially inflate recall here)."""
    proactive_cats = ["C1", "C2", "C3", "C4", "C6"]
    amounts = {r["tx_id"]: float(r["amount_inr"]) for r in dl.transactions}

    tp = fp = tn = fn = 0
    fraud_amount_caught = 0.0
    fraud_amount_missed = 0.0
    fp_review_cost = 0.0
    exceptions = []
    by_posture = defaultdict(lambda: [0, 0])  # label -> [caught, total]

    for tx_id in holdout_ids:
        row = gt[tx_id]
        is_true_fraud = row["label"] == "true_fraud"
        per = results[tx_id]["per_category"]
        flagged = any(per[c]["fired"] for c in proactive_cats)
        amt = amounts.get(tx_id, 0.0)

        if is_true_fraud:
            posture = row.get("fraud_posture", "n/a")
            by_posture[posture][1] += 1
            if flagged:
                by_posture[posture][0] += 1

        if is_true_fraud and flagged:
            tp += 1
            fraud_amount_caught += amt
        elif not is_true_fraud and flagged:
            fp += 1
            fp_review_cost += REVIEW_COST_INR
            if row["label"] in ("friendly_fraud", "merchant_error"):
                # not a Task-A error in the strict sense (these aren't
                # true_fraud), but worth listing as a borderline case
                pass
        elif not is_true_fraud and not flagged:
            tn += 1
        else:  # is_true_fraud and not flagged
            fn += 1
            fraud_amount_missed += amt
            exceptions.append({
                "tx_id": tx_id, "task": "A", "label": row["label"],
                "reason": "true_fraud not caught by any proactive detector",
                "amount_inr": amt,
            })

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    recall_by_posture = {
        k: {"caught": v[0], "total": v[1], "recall": (v[0] / v[1] if v[1] else float("nan"))}
        for k, v in by_posture.items()
    }
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "recall_by_fraud_posture": recall_by_posture,
        "fraud_amount_inr_caught": round(fraud_amount_caught, 2),
        "fraud_amount_inr_missed": round(fraud_amount_missed, 2),
        "false_positive_review_cost_inr": round(fp_review_cost, 2),
        "review_cost_assumption_inr_per_case": REVIEW_COST_INR,
        "exceptions": exceptions,
        "note": (
            "friendly_fraud is intentionally excluded from this task's target: "
            "by construction it has no transaction-time anomaly (the purchase "
            "legitimately happened), so it is not detectable pre-dispute. See "
            "Task B for how it's actually caught."
        ),
    }


def task_b(dl, gt, results, holdout_ids):
    """Dispute-evidence classification: C5's recommended action vs
    correct_action, scored only on transactions where a dispute was filed."""
    disputed = [tx_id for tx_id in holdout_ids if gt[tx_id]["correct_action"] != "NONE"]
    amounts = {r["tx_id"]: float(r["amount_inr"]) for r in dl.transactions}

    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    exceptions = []
    misclassified_amount = 0.0

    for tx_id in disputed:
        row = gt[tx_id]
        want = row["correct_action"]
        want_bucket = "CONTEST" if want == "CONTEST_WITH_EVIDENCE" else "CONCEDE"

        c5 = results[tx_id]["per_category"]["C5"]
        got = recommended_action(c5["trigger"])
        got_bucket = "CONTEST" if got == "CONTEST_WITH_EVIDENCE" else (
            "ESCALATE" if got == "ESCALATE_TO_REVIEW" else "CONCEDE"
        )

        confusion[want_bucket][got_bucket] += 1

        ok = (got_bucket == "CONTEST" and want_bucket == "CONTEST") or \
             (got_bucket in ("CONCEDE", "ESCALATE") and want_bucket == "CONCEDE")
        if ok:
            correct += 1
        else:
            amt = amounts.get(tx_id, 0.0)
            misclassified_amount += amt
            exceptions.append({
                "tx_id": tx_id, "task": "B", "label": row["label"],
                "correct_action": want, "system_recommended": got,
                "amount_inr": amt,
            })

    accuracy = correct / len(disputed) if disputed else float("nan")
    return {
        "n_disputed": len(disputed),
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "misclassified_dispute_amount_inr": round(misclassified_amount, 2),
        "exceptions": exceptions,
    }


def main():
    dl = DataLayer()
    gt = load_ground_truth()
    results = asyncio.run(run_pipeline(dl))
    holdout_ids = [tx_id for tx_id, g in gt.items() if g["split"] == "holdout"]

    report = {
        "holdout_size": len(holdout_ids),
        "task_a_proactive_risk_detection": task_a(dl, gt, results, holdout_ids),
        "task_b_dispute_evidence_classification": task_b(dl, gt, results, holdout_ids),
    }

    print(json.dumps(report, indent=2))

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote eval_report.json", file=sys.stderr)


if __name__ == "__main__":
    main()
