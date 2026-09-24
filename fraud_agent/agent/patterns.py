"""Fraud pattern identification.

Canonical detectors score typologies from signals. Each detector is mapped to a documented
pattern in the bank's typology document (by keyword) when one exists; strong detectors
with no documented counterpart are reported as *undocumented patterns* (the dataset warns
not every pattern is documented). Precedent from similar confirmed cases (memory) is
blended in so analyst-labelled fraud types shape the typing.
"""
from __future__ import annotations

import re

from ..rag.knowledge import get_kb


def _c(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) if hi > lo else 0.0


DETECTORS = {
    "card_testing": {
        "keywords": ["card test", "testing", "probe", "enumerat", "bin attack", "micro", "small value", "low-value", "low value"],
        "score": lambda s: 0.45 * _c(s["small_probe_24h"], 1, 5) + 0.25 * _c(s["burst_1h"], 1, 5)
                         + 0.15 * _c(s["velocity_24h"], 1.0, 2.5) + 0.15 * max(s["new_card"], s["product_c"]),
        "evidence": ["small_probe_24h", "burst_1h", "velocity_24h", "new_card"],
    },
    "account_takeover": {
        "keywords": ["takeover", "account take", "ato", "compromised", "credential", "hijack"],
        "score": lambda s: (0.3 * s["new_device"] + 0.2 * s["new_email"] + 0.2 * s["proxy"] + 0.15 * s["amount_over_max"]
                            + 0.15 * _c(s["amount_z"], 1, 4)) * (1.0 if not s["thin_history"] else 0.4),
        "evidence": ["new_device", "new_email", "proxy", "amount_over_max", "amount_z"],
    },
    "fraud_ring": {
        "keywords": ["ring", "shared device", "collus", "organi", "network of", "linked accounts", "device shar"],
        "score": lambda s: 0.35 * _c(s["shared_device_customers"], 0.7, 2.5) + 0.25 * _c(s["ring_size"], 0.7, 2.5)
                         + 0.25 * _c(s["linked_fraud_customers"] + s["ring_fraud_members"], 0, 3) + 0.15 * _c(s["community_fraud_rate"], 0, 0.3),
        "evidence": ["shared_device_customers", "ring_size", "linked_fraud_customers", "ring_fraud_members"],
    },
    "synthetic_identity": {
        "keywords": ["synthetic", "new account", "fabricated", "identity fraud", "thin file", "bust"],
        "score": lambda s: 0.35 * s["thin_history"] + 0.2 * (1 - _c(s["account_age_days"], 1.5, 4.5)) + 0.2 * s["anon_email"]
                         + 0.25 * _c(s["shared_card_customers"] + s["shared_device_customers"], 0.5, 2),
        "evidence": ["thin_history", "account_age_days", "anon_email", "shared_card_customers"],
    },
    "friendly_fraud": {
        "keywords": ["friendly", "first-party", "first party", "chargeback", "dispute abuse", "false claim", "buyer's remorse"],
        "score": lambda s: s["trigger_customer_report"] * (0.35 * (1 - s["new_device"]) + 0.25 * (1 - s["new_email"])
                         + 0.2 * (1 - s["proxy"]) + 0.2 * (1 - s["thin_history"])),
        "evidence": ["trigger_customer_report", "new_device", "new_email", "proxy"],
    },
    "stolen_card_cnp": {
        "keywords": ["stolen card", "card-not-present", "card not present", "cnp", "lost card", "counterfeit", "compromised card"],
        "score": lambda s: 0.3 * s["new_card"] + 0.2 * s["email_mismatch"] + 0.2 * s["distance_far"] + 0.15 * s["product_c"]
                         + 0.15 * _c(s["model_risk"], 0.3, 0.9),
        "evidence": ["new_card", "email_mismatch", "distance_far", "product_c"],
    },
    "money_mule": {
        "keywords": ["mule", "triangulation", "reship", "pass-through", "layering", "cash out", "cash-out"],
        "score": lambda s: 0.4 * s["email_mismatch"] + 0.3 * _c(s["shared_card_customers"], 0.5, 2) + 0.3 * _c(s["velocity_24h"], 1.5, 3),
        "evidence": ["email_mismatch", "shared_card_customers", "velocity_24h"],
    },
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def documented_map() -> dict[str, str]:
    """canonical detector -> documented pattern name (from the typology doc)."""
    kb = get_kb()
    out = {}
    for canon, d in DETECTORS.items():
        best, best_hits = None, 0
        for name, p in kb.patterns.items():
            text = f"{p['title']} {p['description']}".lower()
            hits = sum(text.count(k) for k in d["keywords"]) + (5 if canon.split("_")[0] in name else 0)
            if hits > best_hits:
                best, best_hits = name, hits
        if best:
            out[canon] = best
    return out


def identify(sig: dict, precedents: list[dict], fraud_prob: float) -> list[dict]:
    dmap = documented_map()
    fraud_prec = [p for p in precedents if p.get("outcome") == "fraud" and p.get("fraud_type")]
    wsum = sum(max(p["score"], 0) for p in fraud_prec) or 1.0
    prec_share: dict[str, float] = {}
    for p in fraud_prec:
        prec_share[_norm(p["fraud_type"])] = prec_share.get(_norm(p["fraud_type"]), 0) + max(p["score"], 0) / wsum

    out = []
    for canon, d in DETECTORS.items():
        det = float(d["score"](sig))
        doc_name = dmap.get(canon)
        prec = max(prec_share.get(_norm(doc_name or ""), 0), prec_share.get(canon, 0))
        score = 0.65 * det + 0.35 * prec if fraud_prec else det
        if canon == "friendly_fraud":   # first-party: the claim is false, i.e. the transaction itself was authorised
            score *= (1 - fraud_prob)
        out.append({"name": doc_name or canon, "canonical": canon, "documented": doc_name is not None,
                    "score": round(score, 3), "detector": round(det, 3), "precedent_support": round(prec, 3),
                    "evidence": {k: round(float(sig.get(k, 0)), 3) for k in d["evidence"]}})
    # precedent-only fraud types (documented types our detectors don't model)
    known = {o["name"] for o in out}
    for ft, share in prec_share.items():
        if ft not in known and share >= 0.3:
            out.append({"name": ft, "canonical": None, "documented": ft in get_kb().patterns, "score": round(0.35 * share, 3),
                        "detector": 0.0, "precedent_support": round(share, 3), "evidence": {}})
    out.sort(key=lambda x: -x["score"])
    return out
