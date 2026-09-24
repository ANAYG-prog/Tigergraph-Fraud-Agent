"""Decision engine: risk, confidence, stopping rule, value-of-information and next best action.

posterior log-odds = calibrated graph/behaviour model (learned from closed cases)
                     + sum(log LR) of additional evidence obtained through controlled actions
confidence         = margin from 0.5 x evidence coverage x agreement with precedent x pattern clarity
enough evidence    = posterior beyond an action threshold AND confidence >= policy minimum
next evidence      = the policy-permitted request with the highest probability of making the
                     decision defensible (expected value of information), net of customer friction
"""
from __future__ import annotations

import math

from .. import settings
from . import signals as S
from .memory import CaseMemory
from .patterns import identify
from .policy import authorize, rules, sar_assessment

COVERAGE_KEYS = ["txn", "profile", "links", "ring", "community", "related_cases"]


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


def evidence_lr(request: str, outcome: str) -> float:
    o = rules()["evidence_requests"].get(request, {}).get("outcomes", {}).get(outcome)
    return (o["pf"] / o["pl"]) if o else 1.0


def voi_rank(p: float, trigger: str, done: set[str], rounds_used: int) -> list[dict]:
    d = rules()["decision"]
    if rounds_used >= d["max_evidence_rounds"]:
        return []
    trig = "customer_report" if "report" in trigger or "customer" in trigger else ("analyst" if "analyst" in trigger else "risk_signal")
    out = []
    for name, spec in rules()["evidence_requests"].items():
        if name in done or trig not in spec.get("applicable_triggers", []):
            continue
        voi, outcomes = 0.0, []
        for o, pr in spec["outcomes"].items():
            po = p * pr["pf"] + (1 - p) * pr["pl"]
            post = p * pr["pf"] / po if po else p
            decisive = post >= d["act_fraud"] or post <= d["act_clear"]
            voi += po * decisive
            outcomes.append({"outcome": o, "prob": round(po, 3), "posterior_if": round(post, 3), "decisive": decisive})
        out.append({"request": name, "description": spec["description"], "p_decisive": round(voi, 3),
                    "friction": spec["friction"], "utility": round(voi - 0.04 * spec["friction"], 3), "outcomes": outcomes})
    return sorted(out, key=lambda x: -x["utility"])


def assess(bundle: dict, mem: CaseMemory, evidence_log: list[dict], exclude_case_ids: set | None = None) -> dict:
    d = rules()["decision"]
    tx = bundle["txn"]
    sig = S.compute(bundle)
    base_lo, contrib = mem.score(sig)
    ev_lo = sum(math.log(e["lr"]) for e in evidence_log if e.get("lr"))
    p0, p = _sig(base_lo), _sig(base_lo + ev_lo)

    similar = mem.similar(sig, k=6, exclude=exclude_case_ids, before_ts=int(tx["ts"]))
    labelled = [s for s in similar if s["outcome"] in ("fraud", "cleared") and not s.get("agent_case")]
    w = sum(max(s["score"], 0) for s in labelled)
    knn_rate = (sum(max(s["score"], 0) for s in labelled if s["outcome"] == "fraud") / w) if w > 0 else None

    patterns = identify(sig, similar, p)
    top = patterns[0] if patterns else None
    clarity = (top["score"] - (patterns[1]["score"] if len(patterns) > 1 else 0)) if top else 0

    report = "report" in (bundle.get("trigger_type") or "") or "customer" in (bundle.get("trigger_type") or "")
    third_party = [x for x in patterns if x["canonical"] != "friendly_fraud"]
    friendly = next((x for x in patterns if x["canonical"] == "friendly_fraud"), None)
    if p >= 0.5:
        likely = (third_party[0]["name"] if third_party and third_party[0]["score"] >= 0.25
                  else "unclassified (possible undocumented pattern)")
    elif report and friendly and friendly["score"] >= 0.3:
        likely = friendly["name"]
    else:
        likely = "none (likely legitimate)"

    coverage = sum(1 for k in COVERAGE_KEYS if bundle.get(k) is not None) / len(COVERAGE_KEYS)
    margin = abs(p - 0.5) * 2
    agreement = 1 - abs(p0 - knn_rate) if knn_rate is not None else 0.5
    ev_strength = min(1.0, abs(ev_lo) / 3)                   # hard evidence from the customer/step-up
    confidence = round(min(1.0, 0.40 * margin + 0.20 * coverage + 0.15 * agreement
                           + 0.10 * min(1, clarity * 3) + 0.15 * ev_strength + 0.0), 3)

    if p >= d["act_fraud"] and confidence >= d["min_confidence"]:
        band, enough = "fraud", True
    elif p <= d["act_clear"] and confidence >= d["min_confidence"]:
        band, enough = "legitimate", True
    else:
        band, enough = "uncertain", False

    done = {e["request"] for e in evidence_log}
    rounds = len(evidence_log)
    voi = [] if enough else voi_rank(p, bundle.get("trigger_type", ""), done, rounds)
    best = voi[0] if voi and voi[0]["p_decisive"] >= 0.25 else None
    if enough:
        stop = f"posterior {p:.2f} beyond {'fraud' if band == 'fraud' else 'clear'} threshold with confidence {confidence:.2f}"
    elif best:
        stop = f"uncertain (p={p:.2f}, confidence={confidence:.2f}); {best['request']} has {best['p_decisive']:.0%} chance of a decisive answer"
    else:
        stop = (f"uncertain (p={p:.2f}, confidence={confidence:.2f}) and no permitted evidence request is likely to resolve it"
                f" (rounds used {rounds}/{d['max_evidence_rounds']}) -> escalate to human")

    related_amt = (bundle.get("ring") or {}).get("amount_30d", 0) if top and top["canonical"] == "fraud_ring" else 0
    sar = sar_assessment(p, float(tx.get("amount") or 0), related_amt, patterns)
    top_name = likely if likely in {x["name"] for x in patterns} else ""
    top_canon = next((x["canonical"] or x["name"] for x in patterns if x["name"] == top_name), "")
    actions = next_best_actions(band, p, confidence, top_canon, sar, tx, bundle.get("trigger_type", ""), best, rounds)

    questions = unresolved_questions(sig, band, report, likely, clarity, done, bundle, patterns)

    return {
        "unresolved_questions": questions,
        "p_fraud": round(p, 4), "p_fraud_graph_only": round(p0, 4), "confidence": round(confidence, 3), "band": band,
        "enough_evidence": enough, "stop_reason": stop,
        "confidence_components": {"margin": round(margin, 3), "coverage": round(coverage, 3), "precedent_agreement": round(agreement, 3),
                                  "pattern_clarity": round(clarity, 3), "additional_evidence_strength": round(ev_strength, 3)},
        "signals": sig, "top_contributions": S.explain(sig, contrib, top=10),
        "evidence_updates": [{"request": e["request"], "outcome": e["outcome"], "likelihood_ratio": round(e["lr"], 2)} for e in evidence_log],
        "patterns": patterns[:5], "likely_fraud_type": likely,
        "similar_cases": similar, "precedent_fraud_rate": None if knn_rate is None else round(knn_rate, 3),
        "next_evidence_options": voi, "recommended_evidence": best, "sar": sar, "recommended_actions": actions,
        "model": {"type": "logistic (fit on closed cases)" if mem.model is not None else "prior weights", **mem.metrics},
    }


def unresolved_questions(sig, band, report, likely, clarity, done, bundle, patterns) -> list[str]:
    """Open questions an analyst would still want answered (drives evidence requests and the explanation)."""
    q = []
    tx = bundle["txn"]
    if not report and "request_customer_validation" not in done and band != "legitimate":
        q.append(f"Did the account holder authorise transaction {tx['txn_id']}? (not yet confirmed with the customer)")
    if report and "request_analyst_info" not in done and band == "uncertain":
        q.append("Does merchant/delivery evidence support or contradict the customer's claim?")
    if sig.get("new_device") and "request_step_up_auth" not in done and band != "legitimate":
        q.append("Is the new device a takeover or a legitimate device change? (step-up not yet performed)")
    if sig.get("shared_device_customers", 0) > 0 and not sig.get("linked_fraud_customers"):
        q.append("Linked customers share this device/network but none has confirmed fraud: collusion, or a shared/public device?")
    if likely.startswith("unclassified"):
        q.append("Activity looks fraudulent but matches no documented pattern: needs analyst typology review")
    elif band != "legitimate" and clarity < 0.1 and len(patterns) > 1:
        q.append(f"Pattern ambiguous between {patterns[0]['name']} and {patterns[1]['name']}")
    for k, err in (bundle.get("errors") or {}).items():
        q.append(f"Evidence source '{k}' failed ({err[:80]}); conclusion made without it")
    return q


def next_best_actions(band, p, conf, top, sar, tx, trigger, best_ev, rounds) -> list[dict]:
    amount = float(tx.get("amount") or 0)
    ctx = {"amount": amount}
    report = "report" in trigger or "customer" in trigger
    acts: list[tuple[str, str]] = [("create_case", "investigation warranted; case record required by policy")]
    if band == "fraud":
        if report:
            acts.append(("refund_customer", "customer-reported unauthorized transaction confirmed; provisional credit (Reg E)"))
        else:
            acts.append(("block_transaction", f"P(fraud)={p:.2f} exceeds action threshold"))
        if top in ("card_testing", "stolen_card_cnp", "account_takeover", "fraud_ring") or not report:
            acts.append(("block_card", f"card credentials likely compromised ({top or 'pattern unclassified'})"))
        if top in ("account_takeover", "fraud_ring", "synthetic_identity", "money_mule") and p >= 0.9:
            acts.append(("block_account", f"{top}: account itself is under attacker control or part of a ring"))
        if top not in ("friendly_fraud", "synthetic_identity", "money_mule"):
            acts.append(("warn_customer", "notify genuine account holder and advise on credential reset"))
        if top == "fraud_ring":
            acts.append(("add_to_watchlist", "flag linked devices/cards/customers from the ring expansion"))
        if sar["required"]:
            acts.append(("file_sar", "; ".join(sar["reasons"])))
        acts.append(("escalate_to_analyst", "approval needed for non-automatic actions"))
    elif band == "legitimate":
        if report and top == "friendly_fraud":
            acts.append(("close_case", "claim not supported: device, email and history consistent with the customer; reject dispute"))
            acts.append(("add_to_watchlist", "possible first-party (friendly) fraud; flag for repeat-dispute monitoring"))
            acts.append(("monitor_account", "watch for further disputes"))
        elif report:
            acts.append(("close_case", "evidence does not support unauthorised use; explain outcome to customer"))
            acts.append(("monitor_account", "residual risk; passive monitoring"))
        else:
            acts.append(("allow_transaction", f"P(fraud)={p:.2f} below clear threshold"))
            acts.append(("close_case", "cleared; record evidence relied upon"))
            if p > 0.05:
                acts.append(("monitor_account", "residual risk; passive monitoring"))
    else:
        if not report and p >= 0.35:
            acts.append(("hold_transaction", f"uncertain (P={p:.2f}); hold pending evidence"))
        acts.append(("monitor_account", "heightened monitoring while uncertain"))
        if best_ev:
            acts.append((best_ev["request"], f"{best_ev['p_decisive']:.0%} chance of a decisive answer; lowest-friction decisive option"))
        else:
            acts.append(("escalate_to_analyst", "evidence rounds exhausted or no decisive request available"))
        if sar["required"]:
            acts.append(("file_sar", "; ".join(sar["reasons"]) + " (pending confirmation)"))
    out, seen = [], set()
    for i, (a, why) in enumerate(acts):
        if a in seen:
            continue
        seen.add(a)
        auth = authorize(a, ctx)
        out.append({"priority": len(out) + 1, "action": a, "rationale": why, "route": auth["route"],
                    "approver": auth["approver"], "execute_now": auth["execute_now"], "policy_note": auth["reason"]})
    return out
