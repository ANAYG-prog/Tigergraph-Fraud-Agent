"""Answer files for the benchmark: one JSON (machine-readable) + one Markdown (human-readable) per case.

`answer()` is the single place that maps the internal case record onto the submission
format; adjust field names here to match the dataset README's answer specification.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import settings


def _nba(decision: dict | None) -> dict | None:
    if not decision:
        return None
    acts = decision["next_best_actions"]
    return {"stage": decision["stage"], "p_fraud": decision["p_fraud"], "confidence": decision["confidence"],
            "assessment": decision["band"],
            "next_best_action": acts[1]["action"] if len(acts) > 1 and acts[0]["action"] == "create_case" else acts[0]["action"],
            "actions": [{"action": a["action"], "approval_route": a["route"], "approver": a["approver"],
                         "auto_executable": a["execute_now"], "rationale": a["rationale"]} for a in acts],
            "recommended_evidence": decision.get("recommended_evidence"),
            "required_approvals": sorted({a["approver"] for a in acts if not a["execute_now"]})}


def meta(case: dict) -> dict:
    closed = next((e["detail"] for e in case["events"] if e["kind"] == "closed_out"), {})
    written = any(e["kind"] == "graph_write" for e in case["events"])
    backend = next((e["detail"].get("backend") for e in case["events"] if e["kind"] == "graph_write"), settings.GRAPH_BACKEND)
    pol = settings.policy_rules().get("meta", {})
    return {
        "dataset": settings.DATASET_LABEL,
        "is_synthetic": settings.IS_SYNTHETIC,
        "answer_format": "provisional - dataset README answer specification not yet available; mapping in fraud_agent/report.py",
        "graph_persistence": ("written to TigerGraph via MCP (FraudCase, CaseEvent, Pattern, SIMILAR_TO, MATCHES_PATTERN)"
                              if written and backend == "tigergraph" else
                              "NOT written to TigerGraph (no configured instance); complete record kept in local outputs only"),
        "policy_config": {"status": pol.get("status", "unknown"), "aligned_with": pol.get("aligned_with")},
        "agent_mode": closed.get("mode"),
        "evidence_sources": sorted({e.get("source", "none") for e in case["evidence_requests"]}),
        "actions": "all executions are simulated; no external system was contacted",
        "disclaimer": ("Model outputs are investigative recommendations, not verified outcomes. "
                       "No accuracy is claimed without an answer key."),
    }


def answer(case: dict) -> dict:
    dec = {d["stage"]: d for d in case["decisions"]}
    before = dec.get("before_evidence_1") or dec.get("initial")
    after_stages = [d for d in case["decisions"] if d["stage"].startswith("after_evidence")]
    after = after_stages[-1] if after_stages else None
    final = dec.get("final")
    return {
        "meta": meta(case),
        "case_id": case["case_id"],
        "trigger": {"type": case["trigger_type"], "detail": case["trigger_detail"], "transaction_id": case["txn_id"],
                    "customer_id": case["customer_id"]},
        "status": case["status"],
        "determination": {"outcome": case["outcome"], "fraud_type": case["fraud_type"],
                          "p_fraud": final and final["p_fraud"], "confidence": final and final["confidence"]},
        "next_best_action": {
            "before_additional_evidence": _nba(before),
            "additional_evidence_requested": [{k: e.get(k) for k in ("request", "justification", "voi", "outcome", "source", "lr",
                                                                     "p_fraud_before")} for e in case["evidence_requests"]],
            "after_additional_evidence": _nba(after) if after else "no additional evidence was required",
            "final": _nba(final),
        },
        "actions_taken": [{"action": a["action"], "system": a["execution"]["system"], "id": a["execution"]["id"]} for a in case["actions_taken"]],
        "pending_approvals": [{"action": a["action"], "approver": a["approver"], "route": a["route"]} for a in case["pending_approvals"]],
        "sar_required": bool(case.get("sar")),
        "sar": case.get("sar"),
        "case_summary": case["summary"],
        "explanation": case["narrative"],
        "unresolved_questions": case.get("unresolved_questions", []),
        "findings": case["findings"],
        "evidence": case["evidence"],
        "evidence_provenance": case.get("provenance", {}),
        "assessments": case["assessments"],
        "investigation_record": case["events"],
        "graph_written": any(e["kind"] == "graph_write" for e in case["events"]),
    }


def _e(x) -> str:
    """escape for markdown table cells (entity keys contain '|')"""
    return str(x).replace("|", "\\|").replace("\n", " ")


def markdown(ans: dict) -> str:
    m = ans["meta"]
    L = [f"# Case {ans['case_id']}", ""]
    if m["is_synthetic"]:
        L += ["> **SYNTHETIC DEVELOPMENT DATA - not an HHGOA benchmark answer.**", ""]
    L += [f"> Dataset: {m['dataset']} · Graph: {m['graph_persistence']} · Policy config: {m['policy_config']['status']} · "
          f"Agent: {m['agent_mode']} · {m['actions']}", "",
         f"**Trigger:** {ans['trigger']['type']} — {ans['trigger']['detail'] or ''}  ",
         f"**Transaction:** {ans['trigger']['transaction_id']} · **Customer:** {ans['trigger']['customer_id']}  ",
         f"**Status:** {ans['status']} · **Outcome:** {ans['determination']['outcome']} · **Type:** {ans['determination']['fraud_type']}  ",
         f"**P(fraud):** {ans['determination']['p_fraud']} · **Confidence:** {ans['determination']['confidence']}", "",
         "## Summary", ans["case_summary"], "", "## Explanation", ans["explanation"], "", "## Next best action"]
    for label, key in (("Before additional evidence", "before_additional_evidence"),
                       ("After additional evidence", "after_additional_evidence"), ("Final", "final")):
        n = ans["next_best_action"][key]
        if isinstance(n, str):
            L += [f"**{label}:** {n}", ""]
            continue
        if not n:
            continue
        L += [f"**{label}** (P={n['p_fraud']}, conf={n['confidence']}, {n['assessment']})", "",
              "| # | Action | Approval route | Approver | Rationale |", "|---|---|---|---|---|"]
        L += [f"| {i + 1} | {a['action']} | {a['approval_route']} | {a['approver']} | {_e(a['rationale'])} |" for i, a in enumerate(n["actions"])]
        L.append("")
    if ans["next_best_action"]["additional_evidence_requested"]:
        L += ["## Additional evidence", "| Request | Why (P decisive) | Outcome | LR | Source |", "|---|---|---|---|---|"]
        L += [f"| {e['request']} | {e['voi']} | {e['outcome']} | {round(e['lr'], 2)} | {e['source']} |"
              for e in ans["next_best_action"]["additional_evidence_requested"]]
        L.append("")
    L += ["## Unresolved questions"] + ([f"- {q}" for q in ans["unresolved_questions"]] or ["- none"]) + [""]
    L += ["## Findings", "| Kind | Finding | Source | Refs |", "|---|---|---|---|"]
    L += [f"| {f.get('kind', '')} | {_e(f['text'])} | {_e(f['source'])} | {_e(', '.join(f.get('refs', []))[:120])} |" for f in ans["findings"]]
    L += ["", "## Evidence provenance", "| Section | Query | Via | Source ids |", "|---|---|---|---|"]
    L += [f"| {k} | {_e(v['query'])} | {v['via']} | {_e(', '.join(v['ids'][:8]))} |" for k, v in ans["evidence_provenance"].items()]
    L.append("")
    if ans["sar"]:
        s = ans["sar"]
        L += ["## Suspicious Activity Report (draft)", f"- Category: {s['part_II_activity']['category']}",
              f"- Amount: ${s['part_II_activity']['amount_involved']:,.2f}", f"- Basis: {'; '.join(s['basis'])}",
              f"- Status: {s['status']}", "", s["part_V_narrative"], ""]
    L += ["## Investigation record", "| # | Actor | Event | Detail |", "|---|---|---|---|"]
    for e in ans["investigation_record"]:
        d = json.dumps(e["detail"], default=str)
        L.append(f"| {e['seq']} | {e['actor']} | {e['kind']} | {d[:220].replace('|', '/')} |")
    return "\n".join(L) + "\n"


def write(case: dict, out_dir: Path | None = None) -> Path:
    d = (out_dir or settings.OUTPUT_DIR) / "answers"
    d.mkdir(parents=True, exist_ok=True)
    ans = answer(case)
    (d / f"{case['case_id']}.json").write_text(json.dumps(ans, indent=2, default=str), encoding="utf-8")
    (d / f"{case['case_id']}.md").write_text(markdown(ans), encoding="utf-8")
    return d / f"{case['case_id']}.json"
