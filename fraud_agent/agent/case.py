"""Fraud case record: status machine, append-only event log, decisions, actions, persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .. import settings

STATUSES = ["new", "investigating", "awaiting_evidence", "pending_approval", "resolved_fraud", "resolved_legitimate",
            "escalated", "closed"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Case:
    case_id: str
    trigger_type: str
    trigger_detail: str
    txn_id: str
    customer_id: str = ""
    opened_ts: int = 0
    status: str = "new"
    is_benchmark: bool = False
    events: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)          # name -> graph evidence summary
    evidence_requests: list = field(default_factory=list)  # controlled requests + responses
    assessments: list = field(default_factory=list)        # [{stage, p_fraud, confidence, ...}]
    decisions: list = field(default_factory=list)          # NBA snapshots (before/after evidence)
    actions_taken: list = field(default_factory=list)      # executed (auto) actions
    pending_approvals: list = field(default_factory=list)  # awaiting human
    findings: list = field(default_factory=list)          # {text, kind: observed|computed|hypothesis, source, refs}
    provenance: dict = field(default_factory=dict)         # evidence section -> {query, via, ids}
    unresolved_questions: list = field(default_factory=list)
    fraud_type: str = ""
    outcome: str = ""
    summary: str = ""
    narrative: str = ""
    sar: dict | None = None
    created_at: str = field(default_factory=now)

    def log(self, kind: str, detail: dict | str, actor: str = "agent"):
        self.events.append({"seq": len(self.events) + 1, "at": now(), "actor": actor, "kind": kind,
                            "detail": detail if isinstance(detail, dict) else {"text": detail}})

    def set_status(self, status: str, why: str):
        if status != self.status:
            self.log("status_change", {"from": self.status, "to": status, "why": why})
            self.status = status

    def add_finding(self, text: str, source: str, kind: str = "computed", refs: list | None = None, actor: str = "agent"):
        f = {"text": text, "kind": kind, "source": source, "refs": [str(r) for r in (refs or []) if r]}
        self.findings.append(f)
        self.log("finding", f, actor=actor)

    @property
    def latest(self) -> dict:
        return self.assessments[-1] if self.assessments else {}

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, out_dir=None):
        d = (out_dir or settings.OUTPUT_DIR) / "cases"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.case_id}.json").write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def load_case(case_id: str, out_dir=None) -> dict | None:
    p = (out_dir or settings.OUTPUT_DIR) / "cases" / f"{case_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
