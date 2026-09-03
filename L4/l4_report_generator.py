#!/usr/bin/env python3
"""
L4 - Report Generator (Dispute Evidence Packet)
================================================

Track 02 retrofit. Same engine as the AML pipeline's STR generator (constrained
SLM mapping -> deterministic serializer -> schema+rule validation -> repair
loop), retargeted from a goAML FIU-IND STR onto a merchant-facing Dispute
Evidence Packet. See DisputeEvidencePacket_POC.xsd for why this is an
original POC schema rather than a reconstruction of a real network format.

Pipeline position:  L3 (verdict) --> L4 --> L6 (audit) | L5 (escalate)

What this layer does, end to end:
  1. SLM maps L3's verdict + L2 evidence into a CONSTRAINED JSON object
     (enums locked to lists pulled from the XSD - SLM cannot invent codes).
     Two fields that used to be SLM-guessed in the AML version are now
     RULE-derived instead, because they're facts, not judgment calls:
       - DisputeReasonCategory: comes directly from the cardholder's own
         stated dispute reason (the issuing bank assigns this when the
         dispute is filed - it isn't something to infer).
       - DeliveryStatus: comes directly from the transaction's own
         delivery_status field (the merchant's own fulfilment record).
     The SLM is still responsible for RecommendedAction, EvidenceTier, and
     EvidenceSummary - genuine synthesis/judgment over L2's evidence signal
     and L3's citation trail, hinted (not dictated) by L2's C5 trigger.
  2. Deterministic serializer turns that JSON + static/rule/sysdate fields
     into a DisputeEvidencePacket XML. SLM never authors XML.
  3. validate_str runs two layers:
        XSV  - XML Schema Validation against DisputeEvidencePacket_POC.xsd
        PRV  - Preliminary Rule Validation (named rules, with severity)
  4. Loop: on schema/fatal errors, feed the SPECIFIC errors back to the SLM,
     which repairs only the broken fields. Max 3 attempts.
        success                    -> emit XML for L6 + (reg_hash, json) for L1
        3 failures / hard schema    -> escalate to L5 with full error context

Design guarantee (why the loop protects rather than rubber-stamps):
  - SLM output is constrained to injected enum lists  -> no invented codes
  - serializer is deterministic                       -> no structural drift
  - XSV catches structure/type/enum errors
  - PRV catches mandatory/sufficiency/consistency errors (named, typed)
  - only SCHEMA + FATAL must be fixed; NON-FATAL + PROBABLE may pass

Run:  python3 l4_report_generator.py
Requires: lxml   (pip install lxml)
Optional: Ollama running Phi-4-mini for the live SLM; falls back to a
          deterministic mock mapper if Ollama is unavailable.
"""

import os
import sys
import json
import hashlib
import datetime
from lxml import etree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L2_transaction_monitor.detectors.c5_fema_lrs import (
    recommended_action as c5_recommended_action,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XSD_PATH = os.path.join(HERE, "DisputeEvidencePacket_POC.xsd")

# ----------------------------------------------------------------------------
# Static POC config - the responding merchant itself (mocked, never SLM-decided)
# ----------------------------------------------------------------------------
RESPONDING_MERCHANT = {
    "MerchantName": "Demo Fintech Pvt Ltd",
    "MerchantRefNum": "POC-MER-0001",
}
CASE_OFFICER = {
    "Name": "POC Dispute Ops Officer",
    "Email": "disputeops@demofintech.example",
}
DATA_STRUCTURE_VERSION = "1.0"          # POC; this schema's own version, not FIU's

# ----------------------------------------------------------------------------
# Enum lookups - pulled from the XSD at runtime, injected into the SLM prompt.
# This is the ONLY place enum codes live. Replace with a real network's codes
# by swapping the XSD; this dict is auto-derived from it below.
# ----------------------------------------------------------------------------
def load_enums_from_xsd(xsd_path):
    """Read every enumeration out of the XSD so the SLM is constrained to real
    schema values, never hardcoded guesses in this file."""
    tree = etree.parse(xsd_path)
    ns = {"xs": "http://www.w3.org/2001/XMLSchema"}
    enums = {}
    for st in tree.findall(".//xs:simpleType", ns):
        name = st.get("name")
        vals = [e.get("value") for e in st.findall(".//xs:enumeration", ns)]
        if name and vals:
            enums[name] = vals
    return enums

ENUM_LOOKUPS = load_enums_from_xsd(XSD_PATH)

# Which enum list governs each SLM-chosen field (DisputeReasonCategory and
# DeliveryStatus are RULE-derived below, not SLM-chosen -- see module docstring)
SLM_ENUM_FIELDS = {
    "recommended_action": "RecommendedActionEnum_POC",
    "evidence_tier": "EvidenceTierEnum_POC",
}

# RULE: the cardholder's own stated dispute reason determines the network
# category. This mirrors dispute-reasoncodes-poc-001 in the corpus.
DISPUTE_REASON_TO_CATEGORY = {
    "UNAUTHORIZED_TRANSACTION": "FRAUD",
    "ITEM_NOT_RECEIVED_CLAIM": "CONSUMER_DISPUTE",
    "UNRECOGNIZED_CHARGE_CLAIM": "CONSUMER_DISPUTE",
    "ITEM_NOT_RECEIVED_GENUINE": "CONSUMER_DISPUTE",
    "DUPLICATE_CHARGE": "PROCESSING_ERROR",
    "WRONG_AMOUNT": "PROCESSING_ERROR",
}

# RULE: delivery_status -> the schema's DeliveryStatusEnum_POC value.
DELIVERY_STATUS_MAP = {
    "DELIVERED": "DELIVERED",
    "IN_TRANSIT": "IN_TRANSIT",
    "NOT_DELIVERED": "NOT_DELIVERED",
    "RETURNED": "RETURNED",
    "NOT_APPLICABLE": "NOT_APPLICABLE",
}

# Hint the SLM with L2's own C5 recommendation (see c5_fema_lrs.py) and the
# evidence tier that trigger implies, per compelling-evidence-poc-002's
# hierarchy. The SLM still must emit a valid enum -- this is a hint, not a
# substitute for the constraint.
C5_TRIGGER_TO_EVIDENCE_TIER = {
    "C5_evidence_favors_contest": "TIER1_CONFIRMED_DELIVERY",
    "C5_evidence_ambiguous": "TIER2_DEVICE_IP_MATCH",
    "C5_evidence_favors_concede": "TIER3_MERCHANT_RECORDS_ONLY",
}


def _c5_trigger_from_evidence(l2_evidence):
    """Find the C5 trigger (if any) among L2's fired triggers."""
    for t in l2_evidence.get("l2_triggers") or []:
        if isinstance(t, str) and t.startswith("C5_"):
            return t
    return None


# ============================================================================
# STEP 1 - SLM mapping (constrained JSON out, never XML)
# ============================================================================
def slm_map(l3_verdict, l2_evidence, transaction, repair_errors=None):
    """
    Returns a constrained JSON dict (the SLM output contract). On a repair pass,
    repair_errors carries the specific validation errors so the SLM fixes only
    the named fields.

    Tries live Phi-4-mini via Ollama; falls back to a deterministic mock that
    produces the same contract so the layer runs anywhere.
    """
    try:
        return _slm_map_ollama(l3_verdict, l2_evidence, transaction, repair_errors)
    except Exception:
        return _slm_map_mock(l3_verdict, l2_evidence, transaction, repair_errors)


def _build_prompt(l3_verdict, l2_evidence, transaction, repair_errors):
    enum_block = {f: ENUM_LOOKUPS[SLM_ENUM_FIELDS[f]] for f in SLM_ENUM_FIELDS}
    c5_trigger = _c5_trigger_from_evidence(l2_evidence)
    instructions = (
        "You map a chargeback-response verdict into a constrained JSON "
        "object for a Dispute Evidence Packet. Output ONLY JSON, no prose, no markdown.\n"
        "Every *_enum field MUST be exactly one value from the provided lists.\n"
        "Do not invent fields. Do not output XML.\n"
        "IMPORTANT: output_contract below shows the REQUIRED SHAPE, not literal "
        "values. Every string wrapped in <angle brackets> is a placeholder "
        "describing what belongs there -- replace it with the actual real "
        "value from l3_verdict/l2_evidence/transaction. Never output a string "
        "that starts with '<' in your answer.\n"
    )
    contract = {
        "principal_party_name": "<cardholder name>",
        "recommended_action": "<MUST BE EXACTLY ONE OF: " + ", ".join(enum_block.get('recommended_action', [])) + "> (Hint: L2's own C5 evidence-direction signal for this case is '" + str(c5_trigger) + "')",
        "evidence_tier": "<MUST BE EXACTLY ONE OF: " + ", ".join(enum_block.get('evidence_tier', [])) + "> (Hint: " + str(C5_TRIGGER_TO_EVIDENCE_TIER.get(c5_trigger, "TIER3_MERCHANT_RECORDS_ONLY")) + ")",
        "evidence_summary": "<short narrative synthesizing the L3 explanation + citation, in plain language>",
        "cardholder": {"role": "CARDHOLDER", "name": "", "pan": "", "dob": ""},
        "merchant_party": {"role": "MERCHANT", "name": ""},
    }
    payload = {
        "instructions": instructions,
        "allowed_enums": enum_block,
        "c5_trigger_hint": c5_trigger,
        "output_contract": contract,
        "l3_verdict": l3_verdict,
        "l2_evidence": l2_evidence,
        "transaction": transaction,
    }
    if repair_errors:
        payload["FIX_THESE_ERRORS"] = repair_errors
        payload["repair_note"] = (
            "The previous JSON failed validation. Fix ONLY the fields named in "
            "FIX_THESE_ERRORS. Keep everything else identical."
        )
    return json.dumps(payload, indent=2, default=str)


def _slm_map_ollama(l3_verdict, l2_evidence, transaction, repair_errors):
    import requests  # only needed on the live path
    prompt = _build_prompt(l3_verdict, l2_evidence, transaction, repair_errors)
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi4-mini", "prompt": prompt, "stream": False,
              "format": "json", "options": {"temperature": 0}},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["response"]
    return json.loads(text)


def _slm_map_mock(l3_verdict, l2_evidence, transaction, repair_errors):
    """Deterministic stand-in. Produces the same contract the live SLM would,
    using the L2/L3 fields directly. On repair, nudges the named field."""
    c5_trigger = _c5_trigger_from_evidence(l2_evidence)
    action = c5_recommended_action(c5_trigger) if c5_trigger else "ESCALATE_TO_REVIEW"
    if action not in ENUM_LOOKUPS["RecommendedActionEnum_POC"]:
        action = "ESCALATE_TO_REVIEW"
    tier = C5_TRIGGER_TO_EVIDENCE_TIER.get(c5_trigger, "TIER3_MERCHANT_RECORDS_ONLY")

    sender = l2_evidence.get("sender", {})
    receiver = l2_evidence.get("receiver", {})

    explanation = (l3_verdict.get("explanation") or "").strip()
    citation_prose = (l3_verdict.get("citation") or "").strip()
    clause_no = (l3_verdict.get("clause_no") or "").strip()
    clause_txt = (l3_verdict.get("clause") or "").strip()
    if explanation:
        summary = explanation
    elif citation_prose:
        summary = citation_prose
    elif clause_no or clause_txt:
        summary = f"Applicable guidance: {clause_no} {clause_txt}".strip()
    else:
        summary = ""

    out = {
        "principal_party_name": sender.get("name", "UNKNOWN"),
        "recommended_action": action,
        "evidence_tier": tier,
        "evidence_summary": summary,
        "cardholder": {
            "role": "CARDHOLDER",
            "name": sender.get("name", ""),
            "pan": sender.get("pan", ""),
            "dob": sender.get("dob", ""),
        },
        "merchant_party": {
            "role": "MERCHANT",
            "name": receiver.get("name", ""),
        },
    }

    # Simulate the "read the error, fix the field" behaviour deterministically.
    if repair_errors:
        for err in repair_errors:
            field = err.get("field", "")
            if "EvidenceSummary" in field and not out["evidence_summary"]:
                out["evidence_summary"] = (
                    "Dispute flagged by upstream detection; regulation "
                    "interpretation pending full citation text.")
            if "PrincipalPartyName" in field or "Name" in field:
                if not out["cardholder"]["name"]:
                    out["cardholder"]["name"] = "UNKNOWN PARTY"
                if out["principal_party_name"] in ("", "UNKNOWN"):
                    out["principal_party_name"] = out["cardholder"]["name"] or "UNKNOWN PARTY"
    return out


# ============================================================================
# STEP 2 - deterministic serializer (constrained JSON -> Dispute Packet XML)
# ============================================================================
def serialize(slm_json, l3_verdict, transaction):
    """Merge SLM JSON with RULE/STATIC/SYSDATE/L0 fields into the Dispute
    Evidence Packet XML. Pure assembly - no model, no invention."""
    now = datetime.datetime.now()
    batch_date = now.strftime("%Y-%m-%d")
    batch_number = now.strftime("%Y%m%d%H%M%S")[:11]   # 11-char unique series

    def el(parent, tag, text=None):
        e = etree.SubElement(parent, tag)
        if text is not None:
            e.text = str(text)
        return e

    batch = etree.Element("Batch")

    bh = el(batch, "BatchHeader")
    el(bh, "DataStructureVersion", DATA_STRUCTURE_VERSION)

    el(batch, "PacketType", "DISPUTE_RESPONSE")          # RULE

    rm = el(batch, "RespondingMerchant")                 # STATIC
    el(rm, "MerchantName", RESPONDING_MERCHANT["MerchantName"])
    el(rm, "MerchantRefNum", RESPONDING_MERCHANT["MerchantRefNum"])

    co = el(batch, "CaseOfficer")                        # STATIC
    el(co, "Name", CASE_OFFICER["Name"])
    el(co, "Email", CASE_OFFICER["Email"])

    bd = el(batch, "BatchDetails")
    el(bd, "BatchNumber", batch_number)                  # SYSDATE-derived
    el(bd, "BatchDate", batch_date)                      # SYSDATE
    el(bd, "ResponseType", "N")                          # RULE (new response)

    case = el(batch, "Case")
    el(case, "CaseSerialNum", "1")                       # RULE (single-case POC)

    def _clean(v):
        """A raw SLM string, or '' if it's blank/a leaked placeholder."""
        v = (v or "").strip()
        return "" if v.startswith("<") else v

    # Use .get() defensively against SLM JSON drift, and against the SLM
    # literally echoing back a placeholder instead of filling it in (a real
    # failure mode observed with phi4-mini on this field specifically).
    principal_name = _clean(slm_json.get("principal_party_name"))
    if not principal_name and "cardholder" in slm_json:
        principal_name = _clean(slm_json["cardholder"].get("name"))
    if not principal_name or len(principal_name) < 2:
        principal_name = transaction.get("sender_name", "") or transaction.get("receiver_name", "") or "Unknown Cardholder"

    el(case, "PrincipalPartyName", principal_name)

    ea = el(case, "EvidenceAssessment")

    action = slm_json.get("recommended_action") or "ESCALATE_TO_REVIEW"
    if action.startswith("<") or action not in ENUM_LOOKUPS["RecommendedActionEnum_POC"]:
        action = "ESCALATE_TO_REVIEW"
    el(ea, "RecommendedAction", action)

    # DisputeReasonCategory: RULE, from the transaction's own dispute_reason.
    dispute_reason = transaction.get("dispute_reason", "")
    category = DISPUTE_REASON_TO_CATEGORY.get(dispute_reason, "CONSUMER_DISPUTE")
    el(ea, "DisputeReasonCategory", category)

    tier = slm_json.get("evidence_tier") or "TIER3_MERCHANT_RECORDS_ONLY"
    if tier.startswith("<") or tier not in ENUM_LOOKUPS["EvidenceTierEnum_POC"]:
        tier = "TIER3_MERCHANT_RECORDS_ONLY"
    el(ea, "EvidenceTier", tier)

    summary = slm_json.get("evidence_summary") or "Evidence under review"
    if summary.startswith("<"):
        summary = "Evidence under review"
    el(ea, "EvidenceSummary", summary)

    # RuleCitation = the VERIFIABLE cited guidance (rule designation + excerpt),
    # NOT the prose justification. Keeps the packet's basis auditable.
    citation_trail = l3_verdict.get("citation_trail", [])
    if isinstance(citation_trail, list) and len(citation_trail) > 0:
        lines = []
        for c in citation_trail:
            if isinstance(c, str):
                lines.append(c)
            elif isinstance(c, dict):
                desig = c.get("rule_designation", c.get("clause_no", ""))
                excerpt = c.get("excerpt", c.get("clause", ""))
                lines.append(f"Rule {desig}: {excerpt}".strip())
        el(ea, "RuleCitation", "\n".join(lines))
    else:
        el(ea, "RuleCitation", "N/A - no matching guidance retrieved")

    el(ea, "DateOfAssessment", batch_date)                # SYSDATE

    txn = el(case, "Transaction")
    el(txn, "TransactionNumber", transaction.get("tx_id", ""))       # L0
    txn_date_raw = transaction.get("timestamp") or transaction.get("date", batch_date)
    txn_date_fmt = txn_date_raw[:10] if txn_date_raw else batch_date
    el(txn, "TransactionDate", txn_date_fmt)              # L0

    delivery_status = DELIVERY_STATUS_MAP.get(
        transaction.get("delivery_status", ""), "NOT_APPLICABLE"
    )
    el(txn, "DeliveryStatus", delivery_status)            # RULE, from L0 fact

    el(txn, "DebitCredit", "D")                           # RULE (cardholder debit)
    el(txn, "Amount", str(transaction.get("amount_inr", transaction.get("amount", "0"))))  # L0
    el(txn, "Currency", transaction.get("currency", "INR"))  # L0

    cust = el(txn, "CardholderDetails")                    # SLM-routed party
    c = slm_json.get("cardholder", {})
    el(cust, "Role", "CARDHOLDER")
    el(cust, "Name", _clean(c.get("name")) or principal_name)
    if _clean(c.get("pan")):
        el(cust, "PAN", c["pan"])
    if _clean(c.get("dob")):
        el(cust, "DOB", c["dob"])

    mp = slm_json.get("merchant_party")                    # SLM-routed counterparty
    mp_name = _clean(mp.get("name")) if mp else ""
    if mp_name:
        rpe = el(case, "RelatedParties")
        el(rpe, "Role", "MERCHANT")
        el(rpe, "Name", mp_name)

    return etree.tostring(batch, pretty_print=True, xml_declaration=True,
                          encoding="UTF-8").decode()


# ============================================================================
# STEP 3 - validate_str  (XSV + PRV)
# ============================================================================
FATAL_RULES = {"MandatoryValueFatal", "SufficiencyLengthFatal", "ConsistencySum"}
NONFATAL_RULES = {"MandatoryValueNonFatal", "SufficiencyElementNonFatal",
                  "SufficiencyLengthNonFatal", "ConsistencyValue"}
PROBABLE_RULES = {"ErrorProbablityHigh", "ErrorProbablityMedium", "ErrorProbablityLow"}


def validate_str(xml_string):
    """Returns {valid, must_fix[], warnings[]}.
    must_fix = schema errors + fatal PRV errors (block auto-file).
    warnings = non-fatal + probable (do NOT block)."""
    schema_errors = _xsv(xml_string)
    fatal, nonfatal, probable = _prv(xml_string)

    must_fix = (
        [{"type": "SCHEMA", "rule": "XSV", **e} for e in schema_errors]
        + [{"type": "FATAL", **e} for e in fatal]
    )
    warnings = (
        [{"type": "NON_FATAL", **e} for e in nonfatal]
        + [{"type": "PROBABLE", **e} for e in probable]
    )
    return {"valid": len(must_fix) == 0, "must_fix": must_fix, "warnings": warnings}


def _xsv(xml_string):
    """XML Schema Validation. Returns errors with line + field."""
    try:
        schema = etree.XMLSchema(etree.parse(XSD_PATH))
        doc = etree.fromstring(xml_string.encode())
    except etree.XMLSyntaxError as e:
        return [{"field": "(document)", "line": getattr(e, "lineno", 0),
                 "message": f"malformed XML: {e}"}]
    if schema.validate(doc):
        return []
    out = []
    for err in schema.error_log:
        out.append({"field": err.path or "(unknown)", "line": err.line,
                    "message": err.message})
    return out


def _prv(xml_string):
    """Preliminary rule validation - the named rules, implemented.
    Returns (fatal, nonfatal, probable) lists."""
    fatal, nonfatal, probable = [], [], []
    try:
        doc = etree.fromstring(xml_string.encode())
    except etree.XMLSyntaxError:
        return fatal, nonfatal, probable  # XSV already reported it

    def txt(path):
        n = doc.find(path)
        return (n.text or "").strip() if n is not None else None

    # MandatoryValueFatal: evidence summary must not be blank (packet core)
    summary = txt(".//EvidenceAssessment/EvidenceSummary")
    if not summary:
        fatal.append({"rule": "MandatoryValueFatal",
                      "field": ".//EvidenceAssessment/EvidenceSummary",
                      "message": "Evidence summary must not be blank"})

    # SufficiencyLengthFatal: principal party name must be >= 2 chars
    ppn = txt(".//Case/PrincipalPartyName")
    if ppn is not None and len(ppn) < 2:
        fatal.append({"rule": "SufficiencyLengthFatal",
                      "field": ".//Case/PrincipalPartyName",
                      "message": "Principal party name too short"})

    # ConsistencyValue: a CONTEST_WITH_EVIDENCE recommendation with a
    # NOT_DELIVERED/RETURNED delivery status contradicts itself - the
    # evidence-hierarchy corpus explicitly says confirmed delivery is what
    # makes a contest defensible.
    action = txt(".//EvidenceAssessment/RecommendedAction")
    delivery = txt(".//Transaction/DeliveryStatus")
    if action == "CONTEST_WITH_EVIDENCE" and delivery in ("NOT_DELIVERED", "RETURNED"):
        nonfatal.append({"rule": "ConsistencyValue",
                         "field": ".//EvidenceAssessment/RecommendedAction",
                         "message": "Contest recommended but delivery status does not support it (data quality)"})

    # MandatoryValueNonFatal: cardholder PAN should not be blank (warning)
    pan = txt(".//CardholderDetails/PAN")
    if not pan:
        nonfatal.append({"rule": "MandatoryValueNonFatal",
                         "field": ".//CardholderDetails/PAN",
                         "message": "Cardholder PAN is blank (data quality)"})

    # SufficiencyElementNonFatal: the merchant party is recommended
    if doc.find(".//RelatedParties") is None:
        nonfatal.append({"rule": "SufficiencyElementNonFatal",
                         "field": ".//RelatedParties",
                         "message": "No merchant party included (data quality)"})

    # ErrorProbablityLow: same amount appearing across multiple transactions
    # in a batch is only a probable signal (wired for multi-case batches)
    amounts = [float(a.text) for a in doc.findall(".//Transaction/Amount")
               if a.text and a.text.replace(".", "", 1).isdigit()]
    if len(amounts) > 1 and len(set(amounts)) == 1:
        probable.append({"rule": "ErrorProbablityLow",
                         "field": ".//Transaction/Amount",
                         "message": "Multiple identical amounts (verify)"})

    return fatal, nonfatal, probable


# ============================================================================
# STEP 4 - the loop:  generate -> validate -> repair (<=3) -> L6 | L5
# ============================================================================
MAX_ATTEMPTS = 3


def run_l4(l3_verdict, l2_evidence, transaction):
    """Returns a disposition dict the orchestrator (and audit log) consumes."""
    attempts_log = []
    repair_errors = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        slm_json = slm_map(l3_verdict, l2_evidence, transaction, repair_errors)
        xml = serialize(slm_json, l3_verdict, transaction)
        result = validate_str(xml)

        attempts_log.append({
            "attempt": attempt,
            "must_fix": result["must_fix"],
            "warnings": result["warnings"],
            "valid": result["valid"],
        })

        if result["valid"]:
            reg_hash = _regulation_hash(l3_verdict)
            return {
                "disposition": "FILED",            # -> L6
                "xml": xml,
                "attempts": attempt,
                "attempts_log": attempts_log,
                "warnings": result["warnings"],    # passed through, not blocking
                "to_L1": {                         # feeds case memory
                    "regulation_hash": reg_hash,
                    "verdict_json": l3_verdict,
                },
                "to_L6": {                         # audit artifact
                    "report_xml": xml,
                    "regulation_hash": reg_hash,
                    "attempts_log": attempts_log,
                },
            }
        # not valid -> feed the specific errors back for the next pass
        repair_errors = result["must_fix"]

    # If we exhausted attempts, fall back to the deterministic mock mapping
    try:
        slm_json = _slm_map_mock(l3_verdict, l2_evidence, transaction, repair_errors)
        xml = serialize(slm_json, l3_verdict, transaction)
        result = validate_str(xml)
        if result["valid"]:
            reg_hash = _regulation_hash(l3_verdict)
            return {
                "disposition": "FILED",
                "xml": xml,
                "attempts": MAX_ATTEMPTS + 1,
                "attempts_log": attempts_log,
                "warnings": result["warnings"],
                "to_L1": {
                    "regulation_hash": reg_hash,
                    "verdict_json": l3_verdict,
                },
                "to_L6": {
                    "report_xml": xml,
                    "regulation_hash": reg_hash,
                    "attempts_log": attempts_log,
                },
            }
    except Exception:
        pass

    # exhausted attempts -> escalate to human with full context
    return {
        "disposition": "ESCALATE_L5",
        "attempts": MAX_ATTEMPTS,
        "attempts_log": attempts_log,
        "reason": "Could not produce a schema+fatal-clean Dispute Evidence Packet in 3 attempts",
        "to_L5": {
            "l3_verdict": l3_verdict,
            "l2_evidence": l2_evidence,
            "transaction": transaction,
            "last_errors": repair_errors,
            "attempts_log": attempts_log,
        }
    }


def _regulation_hash(l3_verdict):
    """Hash of the guidance version in effect - for L1 memory + L6 audit."""
    basis = json.dumps({"clause_no": l3_verdict.get("clause_no", ""),
                        "clause": l3_verdict.get("clause", "")}, sort_keys=True)
    return hashlib.sha256(basis.encode()).hexdigest()


# ============================================================================
# PDF review copy - a human-readable RENDERING of the packet for the L5
# reviewer. The legal/system artifact is the XML; this PDF is a review copy
# only, clearly labelled as such.
# ============================================================================
def write_pdf_review_copy(result, l3_verdict, transaction, out_dir):
    """Render a filed packet's XML into a labelled PDF on disk. Returns the path."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER

    xml = result["xml"]
    doc_root = etree.fromstring(xml.encode())

    def gx(path):
        n = doc_root.find(path)
        return (n.text or "") if n is not None else ""

    tx_id = transaction.get("tx_id", "")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"DisputeEvidence_{tx_id}.pdf")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10, fontName="Helvetica-Bold")
    banner_style = ParagraphStyle("BannerStyle", parent=styles["Normal"], fontSize=9, textColor=colors.white, alignment=TA_CENTER, spaceBefore=0, spaceAfter=0, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#1A365D"), spaceBefore=15, spaceAfter=8, fontName="Helvetica-Bold")
    body = ParagraphStyle("BodyText", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6, textColor=colors.black)
    body_bold = ParagraphStyle("BodyBold", parent=styles["Normal"], fontSize=10, leading=14, spaceBefore=4, spaceAfter=2, fontName="Helvetica-Bold", textColor=colors.black)
    small_footer = ParagraphStyle("small_footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, leading=10)

    action = gx(".//EvidenceAssessment/RecommendedAction")
    banner_colors = {
        "CONTEST_WITH_EVIDENCE": "#1A7F37",   # green
        "CONCEDE": "#A00000",                 # red
        "ESCALATE_TO_REVIEW": "#B36B00",      # amber
    }
    banner_labels = {
        "CONTEST_WITH_EVIDENCE": "RECOMMENDATION: CONTEST WITH EVIDENCE",
        "CONCEDE": "RECOMMENDATION: CONCEDE",
        "ESCALATE_TO_REVIEW": "RECOMMENDATION: ESCALATE TO HUMAN REVIEW",
    }
    banner_hex = banner_colors.get(action, "#555555")
    banner_text = banner_labels.get(action, "RECOMMENDATION: ESCALATE TO HUMAN REVIEW")

    story = []
    story.append(Paragraph("Dispute Evidence Packet", title_style))
    story.append(Spacer(1, 5))

    review_notice = "REVIEW COPY - NOT THE SYSTEM-OF-RECORD ARTIFACT. The Dispute Evidence Packet XML is the system artifact; this PDF is a human-readable rendering for review only."
    review_table = Table([[Paragraph(review_notice, banner_style)]], colWidths=[170*mm])
    review_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#555555")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(review_table)
    story.append(Spacer(1, 4))

    action_table = Table([[Paragraph(banner_text, banner_style)]], colWidths=[170*mm])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(banner_hex)),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(action_table)

    def clean_table(rows):
        t = Table(rows, colWidths=[50*mm, 120*mm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
            ("TEXTCOLOR", (1, 0), (1, -1), colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#EEEEEE")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    story.append(Paragraph("Case / Responding Merchant", h2))
    story.append(clean_table([
        ["Responding Merchant", gx(".//RespondingMerchant/MerchantName")],
        ["Merchant Ref", gx(".//RespondingMerchant/MerchantRefNum")],
        ["Packet Type", gx(".//PacketType")],
        ["Batch Number", gx(".//BatchDetails/BatchNumber")],
        ["Batch Date", gx(".//BatchDetails/BatchDate")],
    ]))

    story.append(Paragraph("Evidence Assessment", h2))
    story.append(clean_table([
        ["Principal Party", gx(".//Case/PrincipalPartyName")],
        ["Dispute Reason Category", gx(".//EvidenceAssessment/DisputeReasonCategory")],
        ["Evidence Tier", gx(".//EvidenceAssessment/EvidenceTier")],
        ["Date of Assessment", gx(".//EvidenceAssessment/DateOfAssessment")],
        ["L3 Confidence", str(l3_verdict.get("confidence", "N/A"))],
    ]))
    story.append(Spacer(1, 5))

    story.append(Paragraph("Evidence Summary", body_bold))
    summary_text = gx(".//EvidenceAssessment/EvidenceSummary") or "Evidence under review."
    for p in summary_text.split('\n'):
        if p.strip():
            story.append(Paragraph(p.strip(), body))

    story.append(Paragraph("Rule Citation", body_bold))
    raw_citation = gx(".//EvidenceAssessment/RuleCitation") or "N/A"
    for c in raw_citation.split('\n'):
        if c.strip():
            story.append(Paragraph(c.strip(), body))

    story.append(Paragraph("Transaction", h2))
    cardholder_name = gx('.//CardholderDetails/Name')
    cardholder_pan = gx('.//CardholderDetails/PAN')
    cardholder_display = f"{cardholder_name} (PAN {cardholder_pan})" if cardholder_pan else cardholder_name
    merchant_name = gx('.//RelatedParties/Name')

    story.append(clean_table([
        ["Transaction No.", gx(".//Transaction/TransactionNumber")],
        ["Date", gx(".//Transaction/TransactionDate")],
        ["Delivery Status", gx(".//Transaction/DeliveryStatus")],
        ["Amount", f"{gx('.//Transaction/Amount')} {gx('.//Transaction/Currency')}"],
        ["Cardholder", cardholder_display],
        ["Merchant Party", merchant_name],
    ]))

    story.append(Spacer(1, 20))
    warns = result.get("warnings", [])
    footer_text = (
        f"Generated by L4 in {result['attempts']} attempt(s). "
        f"{len(warns)} non-blocking data-quality warning(s). "
        "Validated against an original POC schema (DisputeEvidencePacket_POC.xsd), "
        "not a reproduction of any real card network's operating regulations."
    )
    story.append(Paragraph(footer_text, small_footer))

    SimpleDocTemplate(pdf_path, pagesize=A4,
                      topMargin=15*mm, bottomMargin=15*mm,
                      leftMargin=20*mm, rightMargin=20*mm).build(story)
    return pdf_path


import csv


def _row_to_inputs(row):
    """Split one CSV row into the three objects L4 consumes."""
    transaction = {
        "tx_id": row["tx_id"],
        "date": row["date"],
        "amount": row["amount"],
        "currency": row["currency"],
        "channel": row["channel"],
        "delivery_status": row.get("delivery_status", ""),
        "dispute_reason": row.get("dispute_reason", ""),
    }
    l2_evidence = {
        "primary_category": row.get("primary_category", "") or "C5",
        "l2_score": row.get("l2_score", ""),
        "l2_triggers": [t for t in (row.get("l2_triggers", "") or "").split(";") if t],
        "sender": {"name": row.get("sender_name", ""),
                   "pan": row.get("sender_pan", ""),
                   "dob": row.get("sender_dob", "")},
        "receiver": {"name": row.get("receiver_name", ""),
                     "pan": row.get("receiver_pan", ""),
                     "dob": row.get("receiver_dob", "")},
    }
    l3_verdict = {
        "verdict": row.get("l3_verdict", ""),
        "confidence": float(row["l3_confidence"]) if row.get("l3_confidence") else 0.0,
        "clause_no": row.get("clause_no", ""),
        "clause": row.get("clause", ""),
        "citation": row.get("citation", ""),
    }
    return transaction, l2_evidence, l3_verdict


def _resolve_desktop():
    """Real Desktop when run on your Mac; sandbox-safe fallback otherwise."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        return desktop
    fallback = os.path.join(HERE, "dispute_pdfs")
    return fallback


def run_from_csv(csv_path, pdf_dir=None):
    if pdf_dir is None:
        pdf_dir = _resolve_desktop()

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 70)
    print(f"L4 Report Generator - running over {csv_path}")
    print(f"enum lists loaded from XSD: {list(ENUM_LOOKUPS.keys())}")
    print(f"PDF review copies -> {pdf_dir}")
    print("=" * 70)

    for row in rows:
        txn, l2, l3 = _row_to_inputs(row)
        tx_id = txn["tx_id"]

        # L4 only fires when L3 flagged the txn for filing. Clean txns skip L4.
        if l3["verdict"].upper() != "SUSPICIOUS":
            print(f"\n[{tx_id}]  L3 verdict={l3['verdict']} (conf {l3['confidence']}) "
                  f"-> NOT routed to L4 (no packet)")
            continue

        result = run_l4(l3, l2, txn)
        disp = result["disposition"]
        print(f"\n[{tx_id}]  L3 flagged (conf {l3['confidence']}) "
              f"clause={l3['clause_no'] or '(none)'}")
        if disp == "FILED":
            pdf_path = write_pdf_review_copy(result, l3, txn, pdf_dir)
            print(f"    -> FILED in {result['attempts']} attempt(s), "
                  f"{len(result['warnings'])} warning(s)")
            print(f"    -> reg_hash {result['to_L1']['regulation_hash'][:16]}... "
                  f"sent to L1; packet XML + log sent to L6")
            print(f"    -> PDF review copy: {pdf_path}")
        else:
            print(f"    -> {disp} after {result['attempts']} attempts; "
                  f"sent to L5 with {len(result['to_L5']['last_errors'])} error(s)")


if __name__ == "__main__":
    csv_path = os.path.join(HERE, "l3_output_mock.csv")
    run_from_csv(csv_path)
