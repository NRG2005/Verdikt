#!/usr/bin/env python3
"""
generate_chargeback_dataset.py

Builds a synthetic e-commerce/card transaction dataset for the Track 02
("AI Risk Manager" — chargeback evidence responder) retrofit of the
compliance pipeline.

Design goals:
  - Reuses the EXACT CSV filenames + column contract L2_transaction_monitor's
    DataLayer already expects (transactions.csv, account_details.csv,
    watchlist.csv, case_history.csv), so detector code / data_layer.py need
    no structural changes — only new domain columns appended and new content.
  - Ground truth (ground_truth.csv) is a SEPARATE file, keyed by tx_id, that
    no detector/DataLayer/LLM-prompt code path reads. It exists only for the
    eval harness to score against, after the fact.
  - A `split` column (train/holdout) marks which rows are safe to look at
    while tuning thresholds vs which are reserved for honest scoring.
  - Labels are NOT perfectly separable: legitimate edge cases (foreign
    travel, large first-time purchases, dormant-account reactivations that
    are genuine) are deliberately included so precision/recall isn't
    artificially perfect.

Label taxonomy (ground_truth.csv "label" column):
  legitimate     - normal transaction, no dispute
  true_fraud     - card/account was genuinely compromised; dispute (if any)
                   is valid, correct action is to concede
  friendly_fraud - cardholder disputes a transaction that legitimately
                   happened (they have the goods / used the service);
                   correct action is to contest with evidence
  merchant_error - dispute caused by a real merchant-side fault (item never
                   shipped, duplicate charge, wrong amount); correct action
                   is to concede + flag internally, not contest

Run:
  python3 generate_chargeback_dataset.py
"""

import csv
import hashlib
import json
import os
import random
from datetime import datetime, timedelta

random.seed(20260902)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "L2_transaction_monitor", "data")

N_ACCOUNTS = 260          # cardholder ("sender") accounts
N_MERCHANTS = 40          # merchant ("receiver") accounts
N_TRANSACTIONS = 900

FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Krishna","Ishaan","Rohan",
    "Ananya","Diya","Priya","Isha","Kavya","Meera","Neha","Riya","Saanvi","Tanvi",
    "Rahul","Karan","Amit","Sanjay","Vikram","Nikhil","Manish","Suresh","Deepak","Rajesh",
    "Pooja","Sneha","Anjali","Divya","Shreya","Kiran","Nisha","Swati","Preeti","Ritu",
]
LAST_NAMES = [
    "Sharma","Verma","Gupta","Singh","Kumar","Patel","Reddy","Nair","Iyer","Menon",
    "Joshi","Rao","Mehta","Shah","Kapoor","Malhotra","Bose","Chatterjee","Pillai","Desai",
]
CITIES = [
    ("Mumbai","MH","IN",19.0760,72.8777),("Delhi","DL","IN",28.7041,77.1025),
    ("Bengaluru","KA","IN",12.9716,77.5946),("Chennai","TN","IN",13.0827,80.2707),
    ("Hyderabad","TG","IN",17.3850,78.4867),("Pune","MH","IN",18.5204,73.8567),
    ("Kolkata","WB","IN",22.5726,88.3639),("Ahmedabad","GJ","IN",23.0225,72.5714),
    ("Jaipur","RJ","IN",26.9124,75.7873),("Lucknow","UP","IN",26.8467,80.9462),
]
FOREIGN_CITIES = [
    ("Dubai","DU","AE",25.2048,55.2708),("Singapore","SG","SG",1.3521,103.8198),
    ("London","EN","GB",51.5074,-0.1278),("Bangkok","BK","TH",13.7563,100.5018),
]
FATF_CITIES = [
    ("Yangon","YG","MM",16.8409,96.1735),("Panama City","PA","PA",8.9824,-79.5199),
]
CATEGORIES = ["ELECTRONICS","FASHION","GROCERY","TRAVEL","HOME","BEAUTY","SPORTS","DIGITAL_GOODS"]
CHANNELS = ["CARD_ECOM","CARD_POS","UPI_ECOM","WALLET"]

MERCHANT_NAMES = [
    "Nova Electronics","UrbanThreads","QuickCart","HomeNest","BeautyBox","SportFlex",
    "GadgetHub","StyleLoft","FreshMart","TravelEase","DigiStore","PrimeGoods",
]


def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_ts(days_back_max=180, days_back_min=0):
    base = datetime(2026, 9, 1)
    delta = random.uniform(days_back_min, days_back_max)
    return base - timedelta(days=delta, seconds=random.randint(0, 86399))


def rand_ts_daytime(days_back_max=180, days_back_min=0):
    """Same as rand_ts but biased toward realistic e-commerce purchase hours
    (7am-11pm gets 90% weight, overnight 1-5am gets the rest) -- a uniform
    24h distribution isn't realistic and was manufacturing false positives
    on C6's odd-hour signal for perfectly normal shoppers."""
    base = datetime(2026, 9, 1)
    delta = random.uniform(days_back_min, days_back_max)
    if random.random() < 0.90:
        hour = random.randint(7, 22)
    else:
        hour = random.choice([0, 5, 6, 23])
    minute, second = random.randint(0, 59), random.randint(0, 59)
    day_start = base - timedelta(days=delta)
    return day_start.replace(hour=hour, minute=minute, second=second, microsecond=0)


def iso(dt):
    return dt.isoformat()


def device_id(seed):
    return f"DEV-{hashlib.md5(seed.encode()).hexdigest()[:10].upper()}"


def ip_addr(seed):
    h = hashlib.md5(seed.encode()).hexdigest()
    return f"{int(h[0:2],16)}.{int(h[2:4],16)}.{int(h[4:6],16)}.{int(h[6:8],16)}"


# ---------------------------------------------------------------------------
# 1. Accounts (cardholders) + merchants (receivers)
# ---------------------------------------------------------------------------
accounts = []
for i in range(N_ACCOUNTS):
    acc_id = f"ACC{i:05d}"
    city = random.choice(CITIES)
    age_days = random.choice([random.randint(5, 89)] * 2 + [random.randint(90, 2500)] * 8)
    dormancy = 0
    if random.random() < 0.08:
        dormancy = random.randint(150, 500)
    travel_profile = random.choices(
        ["DOMESTIC_STATIC", "DOMESTIC_TRAVELLER", "INTERNATIONAL_FREQUENT"],
        weights=[0.70, 0.20, 0.10],
    )[0]
    accounts.append({
        "account_id": acc_id,
        "pan": f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{random.randint(1000,9999)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",
        "holder_name": rand_name(),
        "dob": f"{random.randint(1965,2003)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "account_age_days": age_days,
        "account_type": "CARD_WALLET",
        "kyc_status": random.choices(["Full KYC", "Min-KYC"], weights=[0.85, 0.15])[0],
        "home_state": city[1],
        "home_city": city[0],
        "home_country": "IN",
        "typical_device_id": device_id(acc_id + "_home"),
        "avg_monthly_txn_count": random.randint(2, 25),
        "avg_monthly_txn_value_inr": round(random.uniform(2000, 60000), 2),
        "avg_tx_amount_inr": round(random.uniform(500, 8000), 2),
        "balance_inr": round(random.uniform(3000, 300000), 2),
        "previous_flags": 0,
        "previous_strs": 0,
        "linked_accounts_count": random.randint(1, 3),
        "occupation_category": random.choice(["Salaried", "Business", "Student", "Retired"]),
        "is_pep": "No",
        "negative_news_flag": "No",
        "account_dormancy_days": dormancy,
        "onboarding_channel": random.choice(["App", "Web", "Branch"]),
        "is_registered_merchant": "False",
        "travel_profile": travel_profile,
        # chargeback-domain extras
        "chargeback_count_180d": 0,   # filled in after tx generation
        "refund_count_180d": 0,
    })
acc_by_id = {a["account_id"]: a for a in accounts}
acc_ids = [a["account_id"] for a in accounts]

merchants = []
for i in range(N_MERCHANTS):
    m_id = f"MER{i:04d}"
    merchants.append({
        "account_id": m_id,
        "name": random.choice(MERCHANT_NAMES) + (f" #{i}" if random.random() < 0.3 else ""),
        "category": random.choice(CATEGORIES),
        "account_age_days": random.randint(200, 3000),
        "is_registered_merchant": "True",
    })
merch_by_id = {m["account_id"]: m for m in merchants}
merch_ids = [m["account_id"] for m in merchants]

# a small set of "collector" accounts used for the C3 fan-in/refund-mule pattern
mule_collector = acc_ids[0]
ring_accounts = acc_ids[1:9]   # feed the collector

# a small set of accounts reused as a stolen-card / device fraud ring for C1/C6
takeover_targets = acc_ids[9:24]

# ---------------------------------------------------------------------------
# 2. Transactions + labels
# ---------------------------------------------------------------------------
transactions = []
ground_truth = []
history_legs = []   # case_history.csv rows (90-day prior activity)

LABELS_WEIGHTS = [
    ("legitimate", 0.78),
    ("true_fraud", 0.07),
    ("friendly_fraud", 0.09),
    ("merchant_error", 0.06),
]


def pick_label():
    r = random.random()
    acc = 0.0
    for lbl, w in LABELS_WEIGHTS:
        acc += w
        if r <= acc:
            return lbl
    return "legitimate"


tx_counter = 0
for i in range(N_TRANSACTIONS):
    tx_counter += 1
    tx_id = f"TX{tx_counter:06d}"
    label = pick_label()

    sender = random.choice(acc_ids)
    merchant = random.choice(merch_ids)
    acc = acc_by_id[sender]
    home_city_tuple = next(c for c in CITIES if c[0] == acc["home_city"])
    ts = rand_ts_daytime()
    order_id = f"ORD{tx_counter:06d}"
    amount = round(random.uniform(400, 15000), 2)
    delivery_status = "DELIVERED"
    dispute_filed = 0
    dispute_reason = ""
    posture = "n/a"
    device = acc["typical_device_id"]
    ip = ip_addr(sender + "_home")
    loc = home_city_tuple
    channel = random.choices(CHANNELS, weights=[0.55, 0.15, 0.20, 0.10])[0]
    is_cross_border = "0"
    fx = ""
    usd_equiv = ""
    refund_flag = 0

    if label == "legitimate":
        # small chance of a benign edge case: foreign travel, or a big
        # one-off purchase, or dormant-but-genuine reactivation -- kept
        # deliberately UN-flagged-worthy so detectors have real negatives
        # to stay precise against.
        if acc["travel_profile"] != "DOMESTIC_STATIC" and random.random() < 0.35:
            loc = random.choice(FOREIGN_CITIES)
            is_cross_border = "1"
            fx = "83.20"
            usd_equiv = round(amount / 83.20, 2)
        elif random.random() < 0.05:
            amount = round(random.uniform(20000, 60000), 2)  # one-off big legit purchase
        delivery_status = random.choices(["DELIVERED", "IN_TRANSIT"], weights=[0.9, 0.1])[0]

    elif label == "true_fraud":
        # Two distinct fraud postures, not one -- real fraud isn't always
        # loud. ~55% "obvious" (multiple strong tells, as before). ~45%
        # "subtle": the fraudster keeps the victim's own device/location/
        # normal hours and relies on the transaction itself blending in.
        # Some subtle cases still drain a large fraction of the account
        # balance (a real, weaker tell C6 can catch); others are a pure
        # blend-in with no anomaly a current detector looks for at all --
        # those are INTENDED to be missed, and show up as honest FNs /
        # exceptions rather than being hidden.
        sender = random.choice(takeover_targets)
        acc = acc_by_id[sender]
        # `loc` was computed for the ORIGINAL pre-reassignment sender above;
        # re-anchor it to the actual (reassigned) account's home city so
        # subtle postures don't get a free, unintended "new location" tell.
        loc = next(c for c in CITIES if c[0] == acc["home_city"])
        is_cross_border = "0"
        fx = ""
        usd_equiv = ""
        dispute_filed = 1
        dispute_reason = "UNAUTHORIZED_TRANSACTION"

        posture = random.choices(["obvious", "subtle_drain", "subtle_blend"], weights=[0.55, 0.20, 0.25])[0]

        if posture == "obvious":
            device = device_id(sender + f"_takeover_{i}")
            ip = ip_addr(f"fraud_ring_ip_{i % 6}")   # small pool of reused fraud IPs
            amount = round(random.uniform(8000, 45000), 2)
            ts = ts.replace(hour=random.choice([1, 2, 3, 4]))
            if random.random() < 0.4:
                loc = random.choice(FATF_CITIES)
                is_cross_border = "1"
                fx = "83.20"
                usd_equiv = round(amount / 83.20, 2)
            delivery_status = random.choices(["NOT_DELIVERED", "DELIVERED"], weights=[0.7, 0.3])[0]

        elif posture == "subtle_drain":
            # victim's own device/location/normal hours -- the only tell is
            # the amount relative to their own balance (C6 balance-drain).
            device = acc["typical_device_id"]
            ip = ip_addr(sender + "_home")
            amount = round(float(acc["balance_inr"]) * random.uniform(0.80, 0.97), 2)
            delivery_status = random.choices(["NOT_DELIVERED", "DELIVERED"], weights=[0.6, 0.4])[0]

        else:  # subtle_blend -- no anomaly any current category looks for
            device = acc["typical_device_id"]
            ip = ip_addr(sender + "_home")
            amount = round(float(acc["avg_tx_amount_inr"]) * random.uniform(1.5, 3.0), 2)
            delivery_status = random.choices(["NOT_DELIVERED", "DELIVERED"], weights=[0.6, 0.4])[0]

    elif label == "friendly_fraud":
        # goods genuinely delivered; cardholder disputes anyway. Delivery
        # confirmation is usually clean (strong contest evidence) but not
        # always -- some cases only have partial/in-transit confirmation,
        # which is what makes this a real classification problem rather
        # than a lookup.
        delivery_status = random.choices(["DELIVERED", "IN_TRANSIT"], weights=[0.85, 0.15])[0]
        dispute_filed = 1
        dispute_reason = random.choice(["ITEM_NOT_RECEIVED_CLAIM", "UNRECOGNIZED_CHARGE_CLAIM"])
        if random.random() < 0.5:
            amount = round(random.uniform(3000, 20000), 2)

    else:  # merchant_error
        delivery_status = random.choices(["NOT_DELIVERED", "RETURNED"], weights=[0.7, 0.3])[0]
        dispute_filed = 1
        dispute_reason = random.choice(["ITEM_NOT_RECEIVED_GENUINE", "DUPLICATE_CHARGE", "WRONG_AMOUNT"])

    row = {
        "tx_id": tx_id,
        "timestamp": iso(ts),
        "channel": channel,
        "amount_inr": amount,
        "sender_account_id": sender,
        "sender_name": acc["holder_name"],
        "sender_pan": acc["pan"],
        "sender_dob": acc["dob"],
        "sender_bank": "DemoBank",
        "sender_ifsc": "DEMO0001234",
        "sender_vpa": f"{sender.lower()}@demobank",
        "receiver_account_id": merchant,
        "receiver_name": merch_by_id[merchant]["name"],
        "receiver_pan": "",
        "receiver_dob": "",
        "receiver_bank": "MerchantBank",
        "receiver_vpa": f"{merchant.lower()}@merchantbank",
        "receiver_state": "MH",
        "receiver_city": "Mumbai",
        "tx_location_city": loc[0],
        "tx_location_state": loc[1],
        "tx_location_country": loc[2],
        "tx_location_lat": loc[3],
        "tx_location_lon": loc[4],
        "device_id": device,
        "purpose_code": merch_by_id[merchant]["category"],
        "is_cross_border": is_cross_border,
        "fx_usd_inr": fx,
        "usd_equiv": usd_equiv,
        "beneficiary_id": merchant,
        "tx_status": "COMPLETED",
        # chargeback-domain extras (appended columns; ignored by existing detectors)
        "order_id": order_id,
        "ip_address": ip,
        "delivery_status": delivery_status,
        "dispute_filed": dispute_filed,
        "dispute_reason": dispute_reason,
    }
    transactions.append(row)

    correct_action = {
        "legitimate": "NONE",
        "true_fraud": "CONCEDE",
        "friendly_fraud": "CONTEST_WITH_EVIDENCE",
        "merchant_error": "CONCEDE_AND_FLAG_INTERNALLY",
    }[label]
    ground_truth.append({
        "tx_id": tx_id,
        "is_fraud_risk": 1 if label in ("true_fraud", "friendly_fraud") else 0,
        "label": label,
        "correct_action": correct_action,
        "fraud_posture": posture,
        "split": "holdout" if (tx_counter % 10 in (0, 1, 2)) else "train",  # ~30% holdout
    })

# ---------------------------------------------------------------------------
# 3. Inject a C3 fan-in / refund-mule ring on top of the base set (a handful
#    of true_fraud txns funnelled into one collector, then a sweep out)
# ---------------------------------------------------------------------------
ring_ts_base = datetime(2026, 8, 20, 10, 0, 0)
for j, acc_id in enumerate(ring_accounts):
    tx_counter += 1
    tx_id = f"TX{tx_counter:06d}"
    ts = ring_ts_base + timedelta(minutes=j * 7)
    acc = acc_by_id[acc_id]
    amount = round(random.uniform(1500, 4500), 2)
    row = {
        "tx_id": tx_id, "timestamp": iso(ts), "channel": "WALLET", "amount_inr": amount,
        "sender_account_id": acc_id, "sender_name": acc["holder_name"], "sender_pan": acc["pan"],
        "sender_dob": acc["dob"], "sender_bank": "DemoBank", "sender_ifsc": "DEMO0001234",
        "sender_vpa": f"{acc_id.lower()}@demobank",
        "receiver_account_id": mule_collector, "receiver_name": acc_by_id[mule_collector]["holder_name"],
        "receiver_pan": "", "receiver_dob": "", "receiver_bank": "DemoBank",
        "receiver_vpa": f"{mule_collector.lower()}@demobank",
        "receiver_state": "MH", "receiver_city": "Mumbai",
        "tx_location_city": "Mumbai", "tx_location_state": "MH", "tx_location_country": "IN",
        "tx_location_lat": 19.0760, "tx_location_lon": 72.8777,
        "device_id": device_id(acc_id + "_ring"), "purpose_code": "DIGITAL_GOODS",
        "is_cross_border": "0", "fx_usd_inr": "", "usd_equiv": "",
        "beneficiary_id": mule_collector, "tx_status": "COMPLETED",
        "order_id": f"ORD{tx_counter:06d}", "ip_address": ip_addr("fraud_ring_ip_0"),
        "delivery_status": "NOT_APPLICABLE", "dispute_filed": 0, "dispute_reason": "",
    }
    transactions.append(row)
    ground_truth.append({
        "tx_id": tx_id, "is_fraud_risk": 1, "label": "true_fraud",
        "correct_action": "CONCEDE", "fraud_posture": "obvious", "split": "holdout" if j % 3 == 0 else "train",
    })

# the sweep-out leg (collector cashes out fast, high preservation ratio)
tx_counter += 1
sweep_id = f"TX{tx_counter:06d}"
sweep_amount = round(sum(0.0 for _ in ring_accounts) or 22000.0, 2)
row = {
    "tx_id": sweep_id, "timestamp": iso(ring_ts_base + timedelta(minutes=90)),
    "channel": "WALLET", "amount_inr": sweep_amount,
    "sender_account_id": mule_collector, "sender_name": acc_by_id[mule_collector]["holder_name"],
    "sender_pan": acc_by_id[mule_collector]["pan"], "sender_dob": acc_by_id[mule_collector]["dob"],
    "sender_bank": "DemoBank", "sender_ifsc": "DEMO0001234",
    "sender_vpa": f"{mule_collector.lower()}@demobank",
    "receiver_account_id": "EXT99999", "receiver_name": "External Cashout",
    "receiver_pan": "", "receiver_dob": "", "receiver_bank": "ExternalBank",
    "receiver_vpa": "cashout@externalbank", "receiver_state": "MH", "receiver_city": "Mumbai",
    "tx_location_city": "Mumbai", "tx_location_state": "MH", "tx_location_country": "IN",
    "tx_location_lat": 19.0760, "tx_location_lon": 72.8777,
    "device_id": device_id(mule_collector + "_sweep"), "purpose_code": "DIGITAL_GOODS",
    "is_cross_border": "0", "fx_usd_inr": "", "usd_equiv": "",
    "beneficiary_id": "EXT99999", "tx_status": "COMPLETED",
    "order_id": f"ORD{tx_counter:06d}", "ip_address": ip_addr("fraud_ring_ip_0"),
    "delivery_status": "NOT_APPLICABLE", "dispute_filed": 0, "dispute_reason": "",
}
transactions.append(row)
ground_truth.append({
    "tx_id": sweep_id, "is_fraud_risk": 1, "label": "true_fraud",
    "correct_action": "CONCEDE", "fraud_posture": "obvious", "split": "train",
})

# ---------------------------------------------------------------------------
# 4. Build case_history.csv (90-day prior legs) + fold in dispute/refund
#    counters onto accounts (for C4/C2/C5-replacement signals)
# ---------------------------------------------------------------------------
chargeback_counts = {a: 0 for a in acc_ids}
refund_counts = {a: 0 for a in acc_ids}
for gt, tx in zip(ground_truth, transactions):
    sid = tx["sender_account_id"]
    if sid not in chargeback_counts:
        continue
    if gt["label"] == "friendly_fraud":
        chargeback_counts[sid] += 1
    if tx.get("delivery_status") in ("RETURNED",) or gt["label"] == "merchant_error":
        refund_counts[sid] += 1

for a in accounts:
    a["chargeback_count_180d"] = chargeback_counts.get(a["account_id"], 0)
    a["refund_count_180d"] = refund_counts.get(a["account_id"], 0)

for acc_id in acc_ids:
    acc = acc_by_id[acc_id]
    n_legs = random.randint(2, 12)
    for _ in range(n_legs):
        ts = rand_ts(days_back_max=90)
        history_legs.append({
            "account_id": acc_id, "timestamp": iso(ts),
            "amount_inr": round(random.uniform(300, 9000), 2),
            "channel": random.choice(CHANNELS),
            "counterparty_id": random.choice(merch_ids),
            "direction": "DEBIT",
            "tx_location_lat": "", "tx_location_lon": "",
            "tx_location_city": "", "tx_location_state": "", "tx_location_country": "",
        })

# ---------------------------------------------------------------------------
# 5. Repurposed "watchlist.csv" -> known fraud-ring device/IP registry (C2)
# ---------------------------------------------------------------------------
watchlist_rows = []
for i in range(6):
    watchlist_rows.append({
        "watchlist_id": f"FR{i:04d}",
        "primary_name": f"fraud_ring_ip_{i}",
        "aliases": "",
        "entity_type": "DEVICE_IP",
        "dob_or_incorp": "",
        "nationality_or_country": "",
        "pan": "",
        "passport": "",
        "cin_or_din": "",
        "national_id_last4": "",
        "last_known_address": ip_addr(f"fraud_ring_ip_{i}"),
        "phone": "",
        "listing_source": "INTERNAL_FRAUD_RING_REGISTRY",
        "reference_number": f"FRING-{i:03d}",
        "reason_narrative": "IP/device previously linked to confirmed account-takeover fraud.",
        "listed_date": "2026-06-01",
        "risk_tier": "CRITICAL",
    })

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


tx_fields = list(transactions[0].keys())
write_csv(os.path.join(DATA_DIR, "transactions.csv"), transactions, tx_fields)

acc_fields = list(accounts[0].keys())
write_csv(os.path.join(DATA_DIR, "account_details.csv"), accounts, acc_fields)

write_csv(os.path.join(DATA_DIR, "watchlist.csv"), watchlist_rows, list(watchlist_rows[0].keys()))

hist_fields = ["account_id","timestamp","amount_inr","channel","counterparty_id","direction",
               "tx_location_lat","tx_location_lon","tx_location_city","tx_location_state","tx_location_country"]
write_csv(os.path.join(DATA_DIR, "case_history.csv"), history_legs, hist_fields)

# ground_truth.csv lives at the repo-level data/ path config.py already
# declares (GROUND_TRUTH_CSV = "data/ground_truth.csv") -- kept structurally
# separate from L2_transaction_monitor/data/ so it's obviously not part of
# the detector-visible dataset.
TOP_DATA_DIR = os.path.join(HERE, "data")
os.makedirs(TOP_DATA_DIR, exist_ok=True)
gt_fields = ["tx_id", "is_fraud_risk", "label", "correct_action", "fraud_posture", "split"]
write_csv(os.path.join(TOP_DATA_DIR, "ground_truth.csv"), ground_truth, gt_fields)

# small manifest for humans / the pitch deck
label_counts = {}
for gt in ground_truth:
    label_counts[gt["label"]] = label_counts.get(gt["label"], 0) + 1
manifest = {
    "n_transactions": len(transactions),
    "n_accounts": N_ACCOUNTS,
    "n_merchants": N_MERCHANTS,
    "label_counts": label_counts,
    "holdout_count": sum(1 for gt in ground_truth if gt["split"] == "holdout"),
    "train_count": sum(1 for gt in ground_truth if gt["split"] == "train"),
    "seed": 20260902,
}
with open(os.path.join(TOP_DATA_DIR, "dataset_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
print(f"\nWrote transactions.csv, account_details.csv, watchlist.csv, case_history.csv to {DATA_DIR}")
print(f"Wrote ground_truth.csv, dataset_manifest.json to {TOP_DATA_DIR}")
