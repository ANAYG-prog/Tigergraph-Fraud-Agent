"""Turn graph evidence into named, explainable fraud signals.

Each signal has a numeric value and a human-readable reason. Signals feed the calibrated
risk model (scoring.py), pattern detection (patterns.py) and the case-similarity vector.
"""
from __future__ import annotations

import math

ANON_EMAILS = {"anonymous.com", "protonmail.com", "mail.com", "guerrillamail.com", "yopmail.com", "tutanota.com"}

# (name, description) – fixed order = feature vector layout
SIGNALS = [
    ("model_risk", "Bank model risk score of the trigger transaction"),
    ("amount_log", "log10 of transaction amount"),
    ("amount_z", "Amount vs customer's own history (z-score, capped)"),
    ("amount_over_max", "Amount exceeds customer's previous maximum"),
    ("new_device", "Device fingerprint never used by this customer before"),
    ("no_identity", "Online transaction with no device/identity record"),
    ("new_email", "Purchaser email domain new for this customer"),
    ("anon_email", "Anonymous / disposable purchaser email domain"),
    ("email_mismatch", "Purchaser and recipient email domains differ"),
    ("new_card", "Card not previously used by this customer"),
    ("new_address", "Billing address new for this customer"),
    ("proxy", "Connection flagged as proxy / anonymiser"),
    ("velocity_24h", "Transactions by this customer in prior 24h (log)"),
    ("small_probe_24h", "Low-value (<$10) probe transactions in prior 24h"),
    ("burst_1h", "Transactions in prior hour"),
    ("thin_history", "Customer has little or no history in the window"),
    ("account_age_days", "Days since first observed transaction (log)"),
    ("shared_device_customers", "Other customers seen on the same device/network (log)"),
    ("shared_card_customers", "Other customers seen on the same card (log)"),
    ("linked_fraud_customers", "Linked customers with confirmed-fraud history"),
    ("linked_cleared_customers", "Linked customers with cleared history"),
    ("ring_size", "Size of shared-entity ring around the customer (log)"),
    ("ring_fraud_members", "Ring members with confirmed-fraud history"),
    ("community_fraud_rate", "Confirmed-fraud share in the customer's WCC community"),
    ("related_case_fraud", "Prior graph-linked cases confirmed as fraud"),
    ("related_case_cleared", "Prior graph-linked cases cleared"),
    ("customer_prior_fraud", "Customer previously confirmed as fraud victim/perpetrator"),
    ("customer_prior_cleared", "Customer previously cleared in an investigation"),
    ("trigger_customer_report", "Investigation triggered by a customer report"),
    ("trigger_analyst", "Investigation triggered by an analyst"),
    ("product_c", "ProductCD = C (highest-risk product family in IEEE-CIS)"),
    ("distance_far", "Billing-to-location distance is unusually large"),
]
SIGNAL_NAMES = [s for s, _ in SIGNALS]
SIGNAL_DESC = dict(SIGNALS)


def _l(x) -> float:
    return math.log1p(max(0.0, float(x or 0)))


def compute(bundle: dict) -> dict[str, float]:
    tx, prof = bundle["txn"], bundle.get("profile") or {}
    links, ring = bundle.get("links") or [], bundle.get("ring") or {}
    comm, related = bundle.get("community") or {}, bundle.get("related_cases") or []
    trig = (bundle.get("trigger_type") or "").lower()
    n = int(prof.get("n_txn") or 0)
    amt = float(tx.get("amount") or 0)
    mean = (prof.get("sum_amt") or 0) / n if n else 0
    var = max(0.0, (prof.get("sum_amt2") or 0) / n - mean ** 2) if n else 0
    sd = math.sqrt(var) if var > 0 else max(mean * 0.5, 1.0)
    has_hist = n >= 2

    def novel(val, seen):
        return float(bool(val) and has_hist and val not in set(seen or []))

    by = {}
    for l in links:
        if not l.get("hub"):
            by.setdefault(l["etype"], []).append(l)
    dev_net = sum(l.get("other_customers", 0) for t in ("Device", "Network") for l in by.get(t, []))
    card = sum(l.get("other_customers", 0) for l in by.get("Card", []))
    fraud_linked = sum(l.get("fraud_customers", 0) for ls in by.values() for l in ls)
    cleared_linked = sum(l.get("cleared_customers", 0) for ls in by.values() for l in ls)
    members = ring.get("members") or []
    first_ts = prof.get("first_ts")
    age = (tx["ts"] - first_ts) / 86400 if first_ts else 0
    proxy = str(tx.get("proxy") or "").upper()
    online = bool(tx.get("device_key") or tx.get("device_type"))
    csize = comm.get("size") or 0
    return {
        "model_risk": float(tx.get("risk_score") or 0),
        "amount_log": math.log10(amt + 1),
        "amount_z": max(-3.0, min(6.0, (amt - mean) / sd)) if has_hist else 0.0,
        "amount_over_max": float(has_hist and amt > float(prof.get("max_amt") or 0) * 1.05),
        "new_device": novel(tx.get("device_key"), prof.get("devices")),
        "no_identity": float(not online and str(tx.get("product")) in ("C", "H", "R", "S")),
        "new_email": novel(tx.get("p_email"), prof.get("emails")),
        "anon_email": float(str(tx.get("p_email") or "").lower() in ANON_EMAILS),
        "email_mismatch": float(bool(tx.get("p_email")) and bool(tx.get("r_email")) and tx.get("p_email") != tx.get("r_email")),
        "new_card": novel(tx.get("card_key"), prof.get("cards")),
        "new_address": novel(tx.get("addr_key"), prof.get("addrs")),
        "proxy": float("PROXY" in proxy and "TRANSPARENT" not in proxy),
        "velocity_24h": _l(prof.get("n_24h")),
        "small_probe_24h": float(prof.get("small_24h") or 0),
        "burst_1h": float(prof.get("n_1h") or 0),
        "thin_history": float(n < 3),
        "account_age_days": _l(age),
        "shared_device_customers": _l(dev_net),
        "shared_card_customers": _l(card),
        "linked_fraud_customers": float(fraud_linked),
        "linked_cleared_customers": float(cleared_linked),
        "ring_size": _l(max(0, (ring.get("ring_size") or 1) - 1)),
        "ring_fraud_members": float(sum(1 for m in members if m.get("prior_fraud"))),
        "community_fraud_rate": (comm.get("fraud_members") or 0) / csize if csize > 1 else 0.0,
        "related_case_fraud": float(sum(1 for c in related if c.get("outcome") == "fraud")),
        "related_case_cleared": float(sum(1 for c in related if c.get("outcome") == "cleared")),
        "customer_prior_fraud": float(any(c.get("outcome") == "fraud" and "same_customer" in (c.get("via") or []) for c in related)),
        "customer_prior_cleared": float(any(c.get("outcome") == "cleared" and "same_customer" in (c.get("via") or []) for c in related)),
        "trigger_customer_report": float("customer" in trig or "report" in trig or "dispute" in trig),
        "trigger_analyst": float("analyst" in trig),
        "product_c": float(str(tx.get("product")) == "C"),
        "distance_far": float((tx.get("dist1") or 0) and float(tx.get("dist1") or 0) > 300),
    }


def vector(sig: dict) -> list[float]:
    return [float(sig.get(k, 0.0)) for k in SIGNAL_NAMES]


def explain(sig: dict, contributions: dict[str, float] | None = None, top: int = 8) -> list[dict]:
    items = []
    for k, v in sig.items():
        c = (contributions or {}).get(k)
        if c is None and not v:
            continue
        items.append({"signal": k, "value": round(v, 3), "meaning": SIGNAL_DESC.get(k, k),
                      "log_odds": None if c is None else round(c, 3)})
    items.sort(key=lambda x: -abs(x["log_odds"] or 0) if contributions else -abs(x["value"]))
    return items[:top]
