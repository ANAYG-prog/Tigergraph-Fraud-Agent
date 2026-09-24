"""Policy gate: permissions, approval routes and SAR obligations.

Every action the agent (or the LLM) proposes passes through `authorize()`. The agent can
only *execute* actions whose route is `auto` (and within auto limits); everything else is
recorded as a recommendation awaiting the named approver.
"""
from __future__ import annotations

from .. import settings

ROUTE_ORDER = ["auto", "analyst", "senior", "compliance", "never"]
APPROVER = {"auto": "agent (policy-permitted)", "analyst": "fraud analyst", "senior": "senior analyst / fraud manager",
            "compliance": "BSA/AML compliance officer", "never": "not permitted"}


def rules() -> dict:
    return settings.policy_rules()


def authorize(action: str, context: dict) -> dict:
    spec = rules()["actions"].get(action)
    if spec is None:
        return {"action": action, "allowed": False, "route": "never", "approver": APPROVER["never"],
                "execute_now": False, "reason": "action not defined in policy"}
    route = spec["route"]
    reason = f"policy route for {action} is '{route}'"
    lim = spec.get("max_amount_auto")
    if route == "auto" and lim is not None and float(context.get("amount") or 0) > lim:
        route, reason = "analyst", f"amount ${context.get('amount'):,.2f} exceeds auto limit ${lim:,.0f}"
    if context.get("override_of_engine") and route == "auto":
        route, reason = "analyst", "LLM proposal deviates from decision engine; human approval required"
    return {"action": action, "allowed": route != "never", "route": route, "approver": APPROVER[route],
            "execute_now": route == "auto", "reversible": spec.get("reversible", True), "reason": reason}


def sar_assessment(p_fraud: float, amount: float, related_amount: float, patterns: list[dict]) -> dict:
    r = rules()["sar"]
    suspicious = p_fraud >= rules()["decision"]["act_fraud"] or p_fraud >= 0.6
    total = amount + (related_amount or 0)
    top = [p["canonical"] or p["name"] for p in patterns[:2] if p["score"] >= 0.45]
    reasons = []
    if suspicious and total >= r["required_if_fraud_and_amount_at_least"]:
        reasons.append(f"suspected fraud with aggregate amount ${total:,.2f} >= ${r['required_if_fraud_and_amount_at_least']:,}")
    if total >= r["required_if_amount_at_least_any"] and p_fraud >= 0.5:
        reasons.append(f"aggregate amount ${total:,.2f} >= ${r['required_if_amount_at_least_any']:,}")
    hit = [t for t in top if t in r.get("required_patterns_any_amount", [])]
    if suspicious and hit:
        reasons.append(f"pattern {hit[0]} requires SAR regardless of amount")
    return {"required": bool(reasons), "reasons": reasons, "aggregate_amount": round(total, 2),
            "deadline_days": r["deadline_days"], "route": "compliance"}
