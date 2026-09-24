"""Analyst console: run investigations, watch the case progress, approve actions, resolve cases.

    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraud_agent import report, settings  # noqa: E402
from fraud_agent.agent import actions as ACT  # noqa: E402
from fraud_agent.agent.case import load_case, now  # noqa: E402
from fraud_agent.agent.evidence import ChainEvidence, InteractiveEvidence, SimulatedEvidence, make_provider  # noqa: E402
from fraud_agent.agent.orchestrator import Agent  # noqa: E402
from fraud_agent.agent.policy import rules  # noqa: E402
from fraud_agent.data.dataset import load  # noqa: E402

st.set_page_config(page_title="Fraud Investigation Agent", page_icon="🕵️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;margin-right:6px}
.auto{background:#d1fae5;color:#065f46}.analyst{background:#fef3c7;color:#92400e}.senior{background:#fee2e2;color:#991b1b}
.compliance{background:#ede9fe;color:#5b21b6}.never{background:#e5e7eb;color:#374151}
.small{font-size:.85rem;color:#6b7280}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading dataset, graph backend and case memory...")
def boot():
    ds = load()
    agent = Agent(evidence=SimulatedEvidence())
    return ds, agent


ds, agent = boot()

# ------------------------------------------------------------------ provenance banners (always visible)
if settings.IS_SYNTHETIC:
    st.error(f"SYNTHETIC development data: {settings.DATASET_LABEL}. Nothing on this screen is an HHGOA benchmark result.")
if agent.be.name != "tigergraph":
    st.warning("TigerGraph not configured: graph queries run on a LOCAL MIRROR of the GSQL queries and cases are NOT persisted "
               "to TigerGraph. Set GRAPH_BACKEND=tigergraph and TG_* variables to use the real graph via MCP.")
if settings.policy_rules().get("meta", {}).get("status") != "aligned_with_dataset":
    st.warning("Policy thresholds/approval routes are placeholder defaults (config/policy_rules.yaml), not yet aligned with the "
               "dataset's bank policy.")
st.caption("All actions are simulated. No messages are sent and no cards, accounts, refunds or reports are touched.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🕵️ Fraud Investigation Agent")
    st.caption(f"Graph: **{agent.be.name}** · Graph {settings.TG_GRAPH}")
    st.caption(f"LLM: **{agent.model if settings.llm_enabled() else 'off (deterministic planner)'}**")
    m = agent.mem.metrics
    if m:
        st.caption(f"Case memory: {m.get('n_cases')} closed cases · CV AUC {m.get('cv_auc', '–')} "
                   f"(bank score alone {m.get('model_risk_only_auc', '–')})")
    use_llm = st.toggle("Use LLM agent", value=settings.llm_enabled(), disabled=not settings.llm_enabled())
    st.divider()
    src = st.radio("Trigger source", ["Benchmark case", "Custom trigger"], horizontal=True)
    if src == "Benchmark case" and not ds.benchmark.empty:
        cid = st.selectbox("Case", ds.benchmark.case_id.tolist())
        row = ds.benchmark[ds.benchmark.case_id == cid].iloc[0].to_dict()
        txn, trig, detail, cust = row["txn_id"], row["trigger"] or "risk_signal", row["trigger_text"], row["customer_id"]
        st.caption(f"{trig} · txn {txn}")
    else:
        cid = st.text_input("Case ID", "ADHOC-001")
        txn = st.text_input("Transaction ID", ds.txns.txn_id.iloc[-1])
        trig = st.selectbox("Trigger type", ["risk_signal", "customer_report", "analyst"])
        detail = st.text_area("Trigger detail", "Customer says they did not make this purchase" if trig == "customer_report" else "")
        cust = ""
    st.divider()
    ev_mode = st.radio("Additional evidence", ["dataset", "simulated", "scripted"], horizontal=True,
                       help="scripted = you decide what the customer / step-up / analyst will answer if asked")
    scripted = {}
    if ev_mode == "scripted":
        for req, spec in rules()["evidence_requests"].items():
            scripted[req] = st.selectbox(req.replace("request_", "").replace("_", " "), list(spec["outcomes"]), key=req)
    run = st.button("▶ Run investigation", type="primary", width="stretch")

# ------------------------------------------------------------------ run
if run:
    if ev_mode == "scripted":
        prov = InteractiveEvidence()
        for r, o in scripted.items():
            prov.set(cid, r, o)
    else:
        prov = make_provider(ev_mode, ds.evidence)
    agent.ev = prov
    agent.use_llm = use_llm
    log = st.status("Investigating…", expanded=True)

    def on_event(kind, d):
        log.write(f"🔧 `{d['tool']}` {json.dumps(d['input'])[:160] if d['input'] else ''}")

    case, inv = agent.investigate(cid, txn, trig, detail, cust, is_benchmark=src == "Benchmark case", on_event=on_event)
    report.write(case.to_dict())
    log.update(label=f"Done: {case.status} · P(fraud)={inv.assessment['p_fraud'] if inv.assessment else '–'}", state="complete")
    st.session_state["case_id"] = cid
    st.session_state["assessment"] = inv.assessment
    st.session_state["policy_hits"] = inv.policy_hits
    st.session_state["bundle"] = {k: inv.bundle.get(k) for k in ("ring", "links", "txn")}

cid_view = st.session_state.get("case_id")
tab_case, tab_port = st.tabs(["Case", "Portfolio (all cases)"])

# ------------------------------------------------------------------ portfolio
with tab_port:
    d = settings.OUTPUT_DIR / "answers"
    rows = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        a = json.loads(f.read_text(encoding="utf-8"))
        nb = a["next_best_action"]
        rows.append({"case": a["case_id"], "trigger": a["trigger"]["type"], "status": a["status"], "outcome": a["determination"]["outcome"],
                     "type": a["determination"]["fraud_type"], "P(fraud)": a["determination"]["p_fraud"],
                     "confidence": a["determination"]["confidence"],
                     "NBA before evidence": (nb["before_additional_evidence"] or {}).get("next_best_action"),
                     "evidence": ", ".join(f"{e['request'].replace('request_', '')}={e['outcome']}" for e in nb["additional_evidence_requested"]),
                     "NBA final": (nb["final"] or {}).get("next_best_action"), "SAR": a["sar_required"],
                     "pending approvals": len(a["pending_approvals"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No cases yet. Run an investigation or `python scripts/run_benchmark.py`.")

# ------------------------------------------------------------------ case view
with tab_case:
    case = load_case(cid_view) if cid_view else None
    if not case:
        st.info("Pick a trigger in the sidebar and run an investigation.")
        st.stop()
    a = st.session_state.get("assessment") or {}
    fin = next((x for x in case["decisions"] if x["stage"] == "final"), case["decisions"][-1] if case["decisions"] else {})
    first = case["decisions"][0] if case["decisions"] else {}

    st.subheader(f"Case {case['case_id']} · {case['status'].replace('_', ' ')}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P(fraud) initial", f"{first.get('p_fraud', 0):.2f}")
    c2.metric("P(fraud) final", f"{fin.get('p_fraud', 0):.2f}", delta=f"{fin.get('p_fraud', 0) - first.get('p_fraud', 0):+.2f}")
    c3.metric("Confidence", f"{fin.get('confidence', 0):.2f}")
    c4.metric("Fraud type", case["fraud_type"] or "–")
    c5.metric("SAR", "required" if case.get("sar") else "not required")
    st.markdown(f"**Summary.** {case['summary']}")
    if case.get("unresolved_questions"):
        st.markdown("**Unresolved questions:** " + " · ".join(case["unresolved_questions"]))

    t1, t2, t3, t4, t5, t6 = st.tabs(["Decision & approvals", "Evidence", "Graph", "Patterns & memory", "Timeline", "SAR / policy"])

    with t1:
        st.markdown("#### Next best action: how it changed as evidence arrived")
        for d in case["decisions"]:
            with st.expander(f"{d['stage']} · P={d['p_fraud']:.2f} · conf={d['confidence']:.2f} · {d['band']}",
                             expanded=d["stage"] in ("final",) or d["stage"].startswith("before_evidence")):
                html = ""
                for x in d["next_best_actions"]:
                    html += (f"<div>{x['priority']}. <b>{x['action']}</b> <span class='pill {x['route']}'>{x['route']}</span>"
                             f"<span class='small'>{x['approver']} — {x['rationale']}</span></div>")
                st.markdown(html, unsafe_allow_html=True)
        if case["evidence_requests"]:
            st.markdown("#### Additional evidence (controlled requests)")
            st.dataframe(pd.DataFrame([{"request": e["request"], "why": e["justification"], "P(decisive)": e["voi"],
                                        "P before": e["p_fraud_before"], "outcome": e["outcome"], "LR": round(e["lr"], 2),
                                        "source": e.get("source")} for e in case["evidence_requests"]]),
                         hide_index=True, width="stretch")
        st.markdown("#### Pending human approvals")
        pend = [p for p in case["pending_approvals"] if p["status"] == "awaiting approval"]
        if not pend:
            st.success("Nothing awaiting approval.")
        for i, p in enumerate(case["pending_approvals"]):
            cols = st.columns([4, 1, 1])
            cols[0].markdown(f"**{p['action']}** <span class='pill {p['route']}'>{p['approver']}</span> — {p['status']}",
                             unsafe_allow_html=True)
            if p["status"] == "awaiting approval":
                if cols[1].button("Approve", key=f"ap{i}"):
                    p["status"] = "approved"; rec = ACT.execute(p["action"], case["case_id"], {"approved_by": "analyst (UI)"})
                    case["actions_taken"].append({**p, "execution": rec})
                    case["events"].append({"seq": len(case["events"]) + 1, "at": now(), "actor": "analyst", "kind": "approval",
                                           "detail": {"action": p["action"], "decision": "approved", "execution": rec["id"]}})
                    (settings.OUTPUT_DIR / "cases" / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2, default=str))
                    report.write(case); agent.be.write_events(case["case_id"], case["events"]); st.rerun()
                if cols[2].button("Reject", key=f"rj{i}"):
                    p["status"] = "rejected"
                    case["events"].append({"seq": len(case["events"]) + 1, "at": now(), "actor": "analyst", "kind": "approval",
                                           "detail": {"action": p["action"], "decision": "rejected"}})
                    (settings.OUTPUT_DIR / "cases" / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2, default=str))
                    report.write(case); agent.be.write_events(case["case_id"], case["events"]); st.rerun()
        st.markdown("#### Resolve case (feeds case memory)")
        cc = st.columns([2, 1])
        verdict = cc[0].radio("Analyst verdict", ["confirmed fraud", "cleared"], horizontal=True)
        if cc[1].button("Resolve & remember"):
            sig = (st.session_state.get("assessment") or {}).get("signals")
            if sig:
                agent.mem.remember({"case_id": case["case_id"], "outcome": "fraud" if verdict == "confirmed fraud" else "cleared",
                                    "fraud_type": case["fraud_type"], "summary": case["summary"], "opened_ts": case["opened_ts"],
                                    "actions_taken": [x["action"] for x in case["actions_taken"]]}, sig, analyst_confirmed=True)
            case["status"] = "closed"
            case["outcome"] = "confirmed_fraud" if verdict == "confirmed fraud" else "cleared"
            case["events"].append({"seq": len(case["events"]) + 1, "at": now(), "actor": "analyst", "kind": "resolution",
                                   "detail": {"verdict": verdict, "memory_updated": bool(sig)}})
            (settings.OUTPUT_DIR / "cases" / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2, default=str))
            report.write(case); agent.be.write_events(case["case_id"], case["events"])
            st.success("Case closed; outcome stored in case memory and the graph."); st.rerun()
        st.markdown("#### Explanation")
        st.write(case["narrative"])

    with t2:
        ev = case["evidence"]
        cA, cB = st.columns(2)
        cA.markdown("**Transaction**"); cA.json(ev.get("transaction", {}), expanded=False)
        cB.markdown("**Customer profile (pre-trigger)**"); cB.json(ev.get("customer_profile", {}), expanded=False)
        st.markdown("**Shared entities**")
        if ev.get("shared_entities"):
            st.dataframe(pd.DataFrame(ev["shared_entities"]), hide_index=True, width="stretch")
        st.markdown("**Findings** (observed = read from data · computed = derived signal · hypothesis = model interpretation)")
        st.dataframe(pd.DataFrame([{"kind": f.get("kind"), "finding": f["text"], "source": f["source"],
                                    "refs": ", ".join(f.get("refs", []))} for f in case["findings"]]),
                     hide_index=True, width="stretch")
        st.markdown("**Evidence provenance** (query and source identifiers for each evidence section)")
        st.dataframe(pd.DataFrame([{"section": k, "query": v["query"], "via": v["via"], "ids": ", ".join(v["ids"][:12])}
                                   for k, v in (case.get("provenance") or {}).items()]), hide_index=True, width="stretch")
        if a.get("top_contributions"):
            st.markdown("**What moved the risk** (log-odds contribution)")
            dfc = pd.DataFrame(a["top_contributions"]).set_index("meaning")[["log_odds"]]
            st.bar_chart(dfc, horizontal=True)

    with t3:
        b = st.session_state.get("bundle") or {}
        tx, links, ring = b.get("txn") or {}, b.get("links") or [], b.get("ring") or {}
        if tx:
            dot = ["graph G {", "rankdir=LR; node [style=filled, fontname=Helvetica, fontsize=10];",
                   f'"C:{tx.get("customer_id")}" [shape=box, fillcolor="#bfdbfe"];',
                   f'"T:{tx["txn_id"]}" [shape=ellipse, fillcolor="#fca5a5", label="txn {tx["txn_id"]}\\n${float(tx["amount"]):,.0f}"];',
                   f'"C:{tx.get("customer_id")}" -- "T:{tx["txn_id"]}";']
            for l in links:
                eid = f'{l["etype"]}:{str(l["eid"])[:28]}'
                color = "#e5e7eb" if l.get("hub") else ("#fde68a" if not l.get("fraud_customers") else "#f87171")
                dot.append(f'"{eid}" [shape=diamond, fillcolor="{color}", label="{eid}\\n{l.get("other_customers", "hub")} others"];')
                dot.append(f'"T:{tx["txn_id"]}" -- "{eid}";')
                for oc in (l.get("customer_ids") or [])[:6]:
                    dot.append(f'"C:{oc}" [shape=box, fillcolor="#e0e7ff"]; "{eid}" -- "C:{oc}";')
            dot.append("}")
            st.graphviz_chart("\n".join(dot))
            st.caption("Red diamond = shared entity touching customers with confirmed fraud; grey = hub (ignored).")
        if ring:
            st.markdown(f"**Ring expansion:** {ring.get('ring_size')} customers via {ring.get('shared_entities')} shared entities; "
                        f"30-day ring spend ${ring.get('amount_30d', 0):,.0f}")
            st.dataframe(pd.DataFrame(ring.get("members") or []), hide_index=True, width="stretch")

    with t4:
        if a.get("patterns"):
            st.markdown("**Pattern hypotheses** (detector ⨉ precedent)")
            st.dataframe(pd.DataFrame([{k: p[k] for k in ("name", "score", "detector", "precedent_support", "documented")} for p in a["patterns"]]),
                         hide_index=True, width="stretch")
        if a.get("similar_cases"):
            st.markdown(f"**Similar prior cases** (precedent fraud rate {a.get('precedent_fraud_rate')})")
            st.dataframe(pd.DataFrame(a["similar_cases"]), hide_index=True, width="stretch")
        if case["evidence"].get("related_cases"):
            st.markdown("**Graph-linked prior cases**")
            st.dataframe(pd.DataFrame(case["evidence"]["related_cases"]), hide_index=True, width="stretch")
        st.json(a.get("confidence_components", {}))

    with t5:
        st.dataframe(pd.DataFrame([{"#": e["seq"], "at": e["at"], "actor": e["actor"], "event": e["kind"],
                                    "detail": json.dumps(e["detail"], default=str)[:300]} for e in case["events"]]),
                     hide_index=True, width="stretch", height=520)

    with t6:
        if case.get("sar"):
            s = case["sar"]
            st.markdown(f"**{s['form']}** · {s['status']} · deadline {s['filing_deadline_days']} days")
            st.markdown(f"Category: **{s['part_II_activity']['category']}** · Amount ${s['part_II_activity']['amount_involved']:,.2f}")
            st.markdown("> " + s["part_V_narrative"])
            st.json(s, expanded=False)
        else:
            st.info("No SAR required under policy for this case.")
        for h in st.session_state.get("policy_hits") or []:
            with st.expander(f"{h['source']} § {h['heading']}"):
                st.write(h["text"])
        md = settings.OUTPUT_DIR / "answers" / f"{case['case_id']}.md"
        if md.exists():
            st.download_button("Download answer file (.md)", md.read_text(encoding="utf-8"), file_name=md.name)
            st.download_button("Download answer file (.json)", md.with_suffix(".json").read_text(encoding="utf-8"),
                               file_name=md.with_suffix(".json").name)
