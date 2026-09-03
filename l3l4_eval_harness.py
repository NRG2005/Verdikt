#!/usr/bin/env python3
"""
Batch-measure L3/L4 decision accuracy against ground_truth.csv, on the
disputed holdout subset (the cases where the LLM-driven legal-reasoning +
report-generation pipeline actually produces a recommended action).

This drives the REAL running API (POST /api/transactions/stream) exactly as
the frontend does, one transaction at a time, and reads the final SSE
"result" event for composite_score (confidence) and l4_recommended_action.

Usage: python3 l3l4_eval.py [--limit N] [--out path.json]
"""
import argparse
import csv
import json
import sys
import time

import requests

API = "http://localhost:8000/api/transactions/stream"

BUCKET = {
    "CONTEST_WITH_EVIDENCE": "CONTEST",
    "CONCEDE": "CONCEDE",
    "CONCEDE_AND_FLAG_INTERNALLY": "CONCEDE",
    "ESCALATE_TO_REVIEW": "ESCALATE",
}


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def build_payload(row):
    payload = dict(row)
    payload["amount_inr"] = float(row["amount_inr"])
    return payload


def run_one(row, timeout=300):
    payload = build_payload(row)
    t0 = time.monotonic()
    result = None
    error = None
    with requests.post(API, json=payload, stream=True, timeout=timeout) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            evt = json.loads(line[len("data: "):])
            if evt.get("type") == "result":
                result = evt["result"]
            elif evt.get("type") == "error":
                error = evt.get("message")
    elapsed = time.monotonic() - t0
    return result, error, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="l3l4_eval_report.json")
    ap.add_argument("--tx-ids", default=None, help="comma-separated tx_ids to run instead of full set")
    args = ap.parse_args()

    gt_rows = load_rows("data/ground_truth.csv")
    gt = {r["tx_id"]: r for r in gt_rows}
    tx_rows = {r["tx_id"]: r for r in load_rows("L2_transaction_monitor/data/transactions.csv")}

    holdout_disputed = [
        tx_id for tx_id, g in gt.items()
        if g["split"] == "holdout" and g["correct_action"] != "NONE"
    ]

    if args.tx_ids:
        holdout_disputed = [t for t in args.tx_ids.split(",") if t in holdout_disputed]
    if args.limit:
        holdout_disputed = holdout_disputed[: args.limit]

    print(f"Running {len(holdout_disputed)} disputed holdout cases through the live pipeline...", file=sys.stderr)

    per_case = []
    confusion = {}
    n_packet_generated = 0
    n_no_packet = 0
    correct_when_packet = 0
    calibration_hits = 0  # confidence>=0.7 AND recommended action correct
    calibration_total_ge07 = 0

    for i, tx_id in enumerate(holdout_disputed):
        row = tx_rows[tx_id]
        g = gt[tx_id]
        want = g["correct_action"]
        want_bucket = BUCKET.get(want, want)

        result, error, elapsed = run_one(row)
        rec = {"tx_id": tx_id, "label": g["label"], "correct_action": want, "elapsed_s": round(elapsed, 1)}

        if error:
            rec["error"] = error
            per_case.append(rec)
            print(f"[{i+1}/{len(holdout_disputed)}] {tx_id}: ERROR ({elapsed:.1f}s) {error[:120]}", file=sys.stderr)
            continue

        confidence = result.get("composite_score")
        l4_action = result.get("l4_recommended_action")
        l4_disp = result.get("l4_disposition")
        rec["confidence"] = confidence
        rec["l4_disposition"] = l4_disp
        rec["l4_recommended_action"] = l4_action

        got_bucket = BUCKET.get(l4_action, l4_action) if l4_action else None

        if l4_action is None:
            n_no_packet += 1
            rec["packet_generated"] = False
        else:
            n_packet_generated += 1
            rec["packet_generated"] = True
            ok = got_bucket == want_bucket
            rec["correct"] = ok
            if ok:
                correct_when_packet += 1
            confusion.setdefault(want_bucket, {}).setdefault(got_bucket, 0)
            confusion[want_bucket][got_bucket] += 1

        if confidence is not None and confidence >= 0.70:
            calibration_total_ge07 += 1
            if l4_action is not None and got_bucket == want_bucket:
                calibration_hits += 1

        per_case.append(rec)
        print(
            f"[{i+1}/{len(holdout_disputed)}] {tx_id}: conf={confidence} action={l4_action} "
            f"want={want} ({elapsed:.1f}s)",
            file=sys.stderr,
        )

    report = {
        "n_cases": len(holdout_disputed),
        "n_packet_generated": n_packet_generated,
        "n_no_packet_routed_to_review": n_no_packet,
        "accuracy_when_packet_generated": (
            correct_when_packet / n_packet_generated if n_packet_generated else None
        ),
        "confusion_matrix_when_packet_generated": confusion,
        "calibration_conf_ge_0.70": {
            "n": calibration_total_ge07,
            "correct": calibration_hits,
            "precision_at_auto_file": (
                calibration_hits / calibration_total_ge07 if calibration_total_ge07 else None
            ),
        },
        "per_case": per_case,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
