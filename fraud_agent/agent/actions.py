"""Mock downstream systems (card platform, CRM, messaging, regulatory filing).

Only called for actions the policy gate marks `execute_now`. Every call is logged to
outputs/action_log.jsonl so the demo shows what would have happened in production.
"""
from __future__ import annotations

import json
import uuid

from .. import settings
from .case import now

SYSTEM = {
    "allow_transaction": "authorization-switch", "hold_transaction": "authorization-switch",
    "block_transaction": "authorization-switch", "block_card": "card-management", "block_account": "core-banking",
    "monitor_account": "fraud-monitoring", "warn_customer": "customer-messaging", "refund_customer": "disputes",
    "request_step_up_auth": "identity-service", "request_customer_validation": "customer-messaging",
    "request_analyst_info": "case-management", "create_case": "case-management", "escalate_to_analyst": "case-management",
    "file_sar": "fincen-filing", "close_case": "case-management", "add_to_watchlist": "fraud-monitoring",
}


def execute(action: str, case_id: str, params: dict | None = None) -> dict:
    rec = {"id": uuid.uuid4().hex[:12], "at": now(), "case_id": case_id, "action": action,
           "system": SYSTEM.get(action, "stub"), "params": params or {}, "simulated": True,
           "status": "simulated - no external system was called"}
    with (settings.OUTPUT_DIR / "action_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
