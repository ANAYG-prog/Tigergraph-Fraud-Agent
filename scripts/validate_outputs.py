"""STRUCTURAL validation of answer files (completeness of the record) - this is NOT a fraud-accuracy check.

    python scripts/validate_outputs.py [answers_dir]

Checks every answer JSON contains the case record, evidence with provenance, typed findings,
decisions, actions, NBA + approval route before and after additional evidence, a SAR when the
engine said one is required, and the provenance/limitation metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraud_agent import settings  # noqa: E402

REQUIRED = ["meta", "case_id", "trigger", "status", "determination", "next_best_action", "actions_taken", "pending_approvals",
            "sar_required", "case_summary", "explanation", "unresolved_questions", "findings", "evidence",
            "evidence_provenance", "assessments", "investigation_record"]


def check(a: dict) -> list[str]:
    errs = [f"missing {k}" for k in REQUIRED if k not in a]
    if errs:
        return errs
    nba = a["next_best_action"]
    for key in ("before_additional_evidence", "final"):
        n = nba.get(key)
        if not isinstance(n, dict) or not n.get("actions"):
            errs.append(f"{key}: no NBA")
        elif any("approval_route" not in x for x in n["actions"]):
            errs.append(f"{key}: action without approval route")
    if nba["additional_evidence_requested"] and not isinstance(nba["after_additional_evidence"], dict):
        errs.append("evidence was requested but no after-evidence NBA recorded")
    if not a["findings"] or any(f.get("kind") not in ("observed", "computed", "hypothesis") for f in a["findings"]):
        errs.append("findings missing or not typed observed/computed/hypothesis")
    if not a["evidence_provenance"]:
        errs.append("no evidence provenance")
    final_sar = any(x["action"] == "file_sar" for x in (nba.get("final") or {}).get("actions", []))
    if final_sar and not a.get("sar"):
        errs.append("file_sar recommended but no SAR draft")
    if not any(e["kind"] == "closed_out" for e in a["investigation_record"]):
        errs.append("investigation record not closed out")
    return errs


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.OUTPUT_DIR / "answers"
    files = sorted(d.glob("*.json"))
    bad = 0
    for f in files:
        a = json.loads(f.read_text(encoding="utf-8"))
        errs = check(a)
        bad += bool(errs)
        tag = "SYNTHETIC" if a.get("meta", {}).get("is_synthetic") else "dataset"
        print(f"{'OK ' if not errs else 'ERR'} {f.stem:12s} [{tag}] {'; '.join(errs)}")
    print(f"\n{len(files) - bad}/{len(files)} answer files structurally complete in {d}")
    print("NOTE: structural completeness only - says nothing about whether any fraud determination is correct.")
    sys.exit(1 if bad or not files else 0)
