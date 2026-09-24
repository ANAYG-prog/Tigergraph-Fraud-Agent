"""Suspicious Activity Report draft (FinCEN SAR structure, simulated filing)."""
from __future__ import annotations

from datetime import datetime, timedelta

from .. import settings

CATEGORY = {
    "card_testing": "Credit/Debit card fraud (card testing)", "account_takeover": "Account takeover",
    "fraud_ring": "Suspected organised fraud ring / collusion", "synthetic_identity": "Identity theft / synthetic identity",
    "friendly_fraud": "First-party fraud (false dispute)", "stolen_card_cnp": "Credit/Debit card fraud (card-not-present)",
    "money_mule": "Money mule / funnel activity",
}


def _date(ts: int) -> str:
    ref = datetime.fromisoformat(settings.datamap().get("reference_date", "2017-12-01"))
    return (ref + timedelta(seconds=int(ts or 0))).strftime("%Y-%m-%d")


def draft(case: dict, a: dict, bundle: dict, narrative: str | None = None) -> dict:
    tx, ring = bundle["txn"], bundle.get("ring") or {}
    top = a["patterns"][0] if a["patterns"] else {}
    cat = CATEGORY.get(top.get("canonical") or "", top.get("name", "Other suspicious activity"))
    subjects = [{"role": "account holder / subject", "customer_id": tx.get("customer_id"),
                 "card": tx.get("card_key"), "email_domain": tx.get("p_email"), "device": tx.get("device_key") or "n/a"}]
    if top.get("canonical") == "fraud_ring":
        for m in (ring.get("members") or [])[:10]:
            if m["customer_id"] != tx.get("customer_id"):
                subjects.append({"role": f"linked ring member (hop {m['depth']})", "customer_id": m["customer_id"]})
    if not narrative:
        ev = "; ".join(f"{c['meaning']} ({c['value']})" for c in a["top_contributions"][:6] if (c["log_odds"] or 0) > 0)
        extra = "; ".join(f"{e['request']} -> {e['outcome']}" for e in a["evidence_updates"]) or "none"
        narrative = (
            f"On {_date(tx['ts'])} a {tx.get('product')} transaction {tx['txn_id']} of ${float(tx['amount']):,.2f} on card "
            f"{tx.get('card_key')} was investigated following a {case['trigger_type']} trigger. Graph investigation of "
            f"transaction history, device/identity, shared-entity links and prior cases indicates {cat.lower()} "
            f"(posterior probability {a['p_fraud']:.2f}, confidence {a['confidence']:.2f}). Key indicators: {ev}. "
            f"Additional evidence obtained: {extra}. Aggregate suspicious amount ${a['sar']['aggregate_amount']:,.2f}. "
            f"Actions: {', '.join(x['action'] for x in a['recommended_actions'])}.")
    return {
        "form": "FinCEN SAR (draft, simulated filing)", "case_id": case["case_id"], "filing_type": "initial",
        "status": "draft - pending compliance approval", "filing_deadline_days": a["sar"]["deadline_days"],
        "part_I_subjects": subjects,
        "part_II_activity": {"date_range": f"{_date(tx['ts'] - 30 * 86400)} to {_date(tx['ts'])}",
                             "amount_involved": a["sar"]["aggregate_amount"], "category": cat,
                             "instruments": ["credit/debit card"], "product_type": tx.get("product")},
        "part_IV_filer": {"institution": "HHGOA Bank (simulated)", "contact": "BSA/AML Compliance"},
        "part_V_narrative": narrative, "basis": a["sar"]["reasons"],
    }
