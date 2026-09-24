"""The investigation agent.

The LLM drives the investigation by choosing tools (graph queries via TigerGraph MCP,
GraphRAG over policy/typologies, case memory, controlled evidence requests). Deterministic
components keep it honest: a calibrated decision engine scores risk and confidence, a
policy gate decides what may be executed and who must approve, and a completeness check
fills any mandatory step the LLM skipped. Without an API key the same tools run on a
fixed plan (deterministic mode), producing the same case record without the LLM narrative.
"""
from __future__ import annotations

import json
import time
import traceback

from .. import settings
from ..graph.backend import GraphBackend, get_backend
from ..rag.knowledge import KnowledgeBase, get_kb
from . import actions as ACT
from . import sar as SAR
from .assess import assess, evidence_lr
from .case import Case
from .evidence import EvidenceProvider, make_provider
from .investigate import gather
from .memory import CaseMemory, get_memory
from .policy import authorize, rules
from .signals import vector as signal_vector

MAX_TURNS = 24

SYSTEM_PROMPT = """You are a fraud investigation agent at a bank. You investigate one case at a time using tools \
backed by a TigerGraph knowledge graph (transactions, customers, cards, devices, networks, emails, addresses, prior \
fraud cases), a GraphRAG index over the bank's fraud policy, known fraud patterns and regulations, and case memory of \
prior investigations with analyst outcomes.

How to work:
1. Start from the trigger. Pull the transaction, the customer's behavioural profile, shared-entity links, the ring \
expansion / community, graph-linked prior cases and similar prior cases. Skip nothing that could change the decision.
2. Call assess_risk to get the calibrated posterior P(fraud), confidence, pattern hypotheses, the policy's SAR test \
and the decision engine's next-best actions. The engine's numbers are the source of truth for risk; your job is to \
interpret the evidence, notice what the numbers miss (e.g. an undocumented pattern, a contradiction), and explain.
3. Ground every policy statement in search_policy results and cite the source heading.
4. If assess_risk says the evidence is not enough, request the recommended additional evidence with \
request_evidence (it is policy-gated), then call assess_risk again. Do not request evidence when the assessment is \
already decisive. Never ask the customer for evidence after a customer-initiated report if policy says it is not applicable.
5. Record material findings with record_finding as you go (one fact per finding, with its source tool).
6. Finish with finalize_case exactly once. Keep the engine's recommended actions unless you have a concrete, \
evidence-based reason; any action you add that is stronger than the engine's recommendation is routed to a human.

Be concise and factual. Distinguish evidence from inference. State remaining uncertainty explicitly."""


def _tool(name, desc, props=None, required=None):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}}


TOOLS = [
    _tool("get_transaction", "Trigger transaction with its card, device, network, email and address entities (GSQL txn_context)."),
    _tool("get_customer_profile", "Customer's behavioural baseline before the trigger: counts, amounts, velocity, known devices/emails/cards/addresses (GSQL customer_profile) and the recent timeline."),
    _tool("get_shared_entity_links", "Other customers sharing the trigger's device/network/card/address/email, their risk and confirmed-fraud history; hub entities are flagged (GSQL entity_links)."),
    _tool("expand_fraud_ring", "Bounded BFS over shared non-hub devices/networks/cards from the customer, plus WCC community stats (GSQL ring_expand, community_stats).",
          {"hops": {"type": "integer", "minimum": 1, "maximum": 4, "description": "customer hops (default 3)"}}),
    _tool("get_related_cases", "Prior fraud cases connected in the graph to the same customer or shared entities (GSQL related_cases)."),
    _tool("find_similar_cases", "Nearest prior investigations by evidence-vector similarity, with analyst outcome, fraud type and actions (case memory)."),
    _tool("search_policy", "GraphRAG over the bank fraud policy, documented fraud patterns and regulatory references. Returns cited chunks.",
          {"query": {"type": "string"}, "patterns": {"type": "array", "items": {"type": "string"}, "description": "pattern names to expand via the graph"}},
          ["query"]),
    _tool("assess_risk", "Run the calibrated decision engine on everything gathered so far: P(fraud), confidence, patterns, SAR test, stop rule, value-of-information ranking and policy-routed next-best actions. Call after gathering and after each new evidence."),
    _tool("request_evidence", "Policy-gated controlled evidence request. Records the next-best action BEFORE the request, then returns the response.",
          {"request": {"type": "string", "enum": ["request_customer_validation", "request_step_up_auth", "request_analyst_info"]},
           "justification": {"type": "string"}}, ["request", "justification"]),
    _tool("record_finding", "Add one finding to the case record. Classify it honestly: 'observed' = a fact read directly from the "
          "graph/dataset; 'computed' = a derived signal/score from a tool; 'hypothesis' = your interpretation.",
          {"text": {"type": "string"}, "kind": {"type": "string", "enum": ["observed", "computed", "hypothesis"]},
           "source": {"type": "string", "description": "tool or document the finding comes from"},
           "refs": {"type": "array", "items": {"type": "string"}, "description": "source identifiers: transaction/customer/entity/case/policy chunk ids"}},
          ["text", "kind", "source"]),
    _tool("finalize_case", "Close out the investigation: fraud type, case summary, explanation, and the action list.",
          {"fraud_type": {"type": "string", "description": "pattern name, 'none (legitimate)' or 'undetermined'"},
           "summary": {"type": "string", "description": "3-6 sentence case summary for the case record"},
           "explanation": {"type": "string", "description": "evidence used, why evidence was requested, why the actions, remaining uncertainty"},
           "sar_narrative": {"type": "string", "description": "SAR Part V narrative (who/what/when/where/why/how) if a SAR is required, else empty"},
           "actions": {"type": "array", "items": {"type": "string"}, "description": "final action names (engine's list unless justified)"},
           "deviations": {"type": "string", "description": "why any action differs from the engine's recommendation, else empty"}},
          ["fraud_type", "summary", "explanation", "actions"]),
]


class Investigation:
    """Holds state for one case; tool methods mutate it and return compact JSON for the LLM."""

    def __init__(self, case: Case, backend: GraphBackend, mem: CaseMemory, kb: KnowledgeBase, ev: EvidenceProvider):
        self.case, self.be, self.mem, self.kb, self.ev = case, backend, mem, kb, ev
        self.bundle: dict = {"trigger_type": case.trigger_type, "errors": {}, "timings": {}}
        self.evidence_log: list[dict] = []
        self.assessment: dict | None = None
        self.assessed_since_evidence = False
        self.final: dict | None = None
        self.policy_hits: list[dict] = []

    # ---------------------------------------------------------------- graph tools
    def _prov(self, section, query, ids):
        via = ("TigerGraph MCP (tigergraph__run_installed_query)" if self.be.name == "tigergraph"
               else "LOCAL MIRROR of the GSQL query (TigerGraph not configured)")
        self.case.provenance[section] = {"query": query, "via": via, "ids": [str(i) for i in ids if i][:40]}

    def _gather(self, key, fn):
        t0 = time.perf_counter()
        try:
            self.bundle[key] = fn()
        except Exception as e:
            self.bundle[key] = None
            self.bundle["errors"][key] = str(e)[:300]
        self.bundle["timings"][key] = round(time.perf_counter() - t0, 3)
        return self.bundle[key]

    def get_transaction(self):
        tx = self._gather("txn", lambda: self.be.txn(self.case.txn_id))
        if not tx:
            return {"error": self.bundle["errors"].get("txn")}
        tx["customer_id"] = tx.get("customer_id") or self.case.customer_id
        self.case.customer_id, self.case.opened_ts = tx["customer_id"], int(tx["ts"])
        self.case.evidence["transaction"] = {k: tx.get(k) for k in ("txn_id", "ts", "amount", "product", "risk_score", "customer_id",
                                             "card_key", "addr_key", "p_email", "r_email", "device_key", "device_type", "net_key", "proxy", "dist1")}
        self._prov("transaction", "txn_context", [tx["txn_id"], tx.get("customer_id"), tx.get("card_key"), tx.get("device_key"),
                                                   tx.get("net_key"), tx.get("addr_key"), tx.get("p_email")])
        self.case.log("tool", {"tool": "get_transaction", "query": "txn_context"})
        return self.case.evidence["transaction"]

    def _need_tx(self):
        if not self.bundle.get("txn"):
            self.get_transaction()
        return self.bundle["txn"]

    def get_customer_profile(self):
        tx = self._need_tx()
        p = self._gather("profile", lambda: self.be.profile(tx["customer_id"], int(tx["ts"])))
        recent = self._gather("recent", lambda: self.be.recent(tx["customer_id"], int(tx["ts"]), 15))
        if p is None:
            return {"error": self.bundle["errors"].get("profile")}
        n = p.get("n_txn") or 0
        s = {"n_txn_180d": n, "mean_amount": round((p.get("sum_amt") or 0) / n, 2) if n else None, "max_amount": p.get("max_amt"),
             "txn_last_24h": p.get("n_24h"), "txn_last_1h": p.get("n_1h"), "small_txn_last_24h": p.get("small_24h"),
             "first_seen_days_before": round((tx["ts"] - p["first_ts"]) / 86400, 1) if p.get("first_ts") else None,
             "known_devices": len(p.get("devices") or []), "known_emails": p.get("emails"), "known_cards": len(p.get("cards") or []),
             "trigger_device_known": (tx.get("device_key") in (p.get("devices") or [])) if tx.get("device_key") else None,
             "trigger_email_known": tx.get("p_email") in (p.get("emails") or []),
             "trigger_card_known": tx.get("card_key") in (p.get("cards") or []),
             "recent_txns": [{k: r.get(k) for k in ("txn_id", "ts", "amount", "product", "risk_score")} for r in (recent or [])[:10]]}
        self.case.evidence["customer_profile"] = s
        self._prov("customer_profile", "customer_profile, recent_txns", [tx["customer_id"]] + [r.get("txn_id") for r in (recent or [])[:10]])
        self.case.log("tool", {"tool": "get_customer_profile", "query": "customer_profile, recent_txns"})
        return s

    def get_shared_entity_links(self):
        self._need_tx()
        links = self._gather("links", lambda: self.be.entity_links(self.case.txn_id))
        if links is None:
            return {"error": self.bundle["errors"].get("links")}
        s = [{k: l.get(k) for k in ("etype", "eid", "degree", "hub", "other_customers", "other_txns", "avg_risk",
                                    "fraud_customers", "cleared_customers")} for l in links]
        self.case.evidence["shared_entities"] = s
        self._prov("shared_entities", "entity_links", [f"{l['etype']}:{l['eid']}" for l in links]
                   + [c for l in links for c in (l.get("customer_ids") or [])[:5]])
        self.case.log("tool", {"tool": "get_shared_entity_links", "query": "entity_links"})
        return s

    def expand_fraud_ring(self, hops=3):
        tx = self._need_tx()
        ring = self._gather("ring", lambda: self.be.ring(tx["customer_id"], int(tx["ts"]), int(hops or 3)))
        comm = self._gather("community", lambda: self.be.community(tx["customer_id"]))
        if ring is None:
            return {"error": self.bundle["errors"].get("ring")}
        mem = ring.get("members") or []
        s = {"ring_size": ring.get("ring_size"), "shared_entities": ring.get("shared_entities"),
             "ring_amount_30d": round(ring.get("amount_30d") or 0, 2), "ring_txns_30d": ring.get("txns_30d"),
             "ring_avg_risk_30d": round(ring.get("avg_risk_30d") or 0, 3),
             "members_with_confirmed_fraud": sum(1 for m in mem if m.get("prior_fraud")),
             "members_cleared": sum(1 for m in mem if m.get("prior_cleared")),
             "members_sample": [m["customer_id"] for m in mem[:12]], "wcc_community": comm}
        self.case.evidence["ring"] = s
        self._prov("ring", "ring_expand, community_stats", [m["customer_id"] for m in mem[:40]])
        self.case.log("tool", {"tool": "expand_fraud_ring", "query": "ring_expand, community_stats", "hops": hops})
        return s

    def get_related_cases(self):
        self._need_tx()
        rel = self._gather("related_cases", lambda: self.be.related_cases(self.case.txn_id))
        if rel is None:
            return {"error": self.bundle["errors"].get("related_cases")}
        self.case.evidence["related_cases"] = rel[:15]
        self._prov("related_cases", "related_cases", [c["case_id"] for c in rel[:40]])
        self.case.log("tool", {"tool": "get_related_cases", "query": "related_cases", "n": len(rel)})
        return rel[:15]

    def find_similar_cases(self):
        a = self._assess(silent=True)
        self.case.evidence["similar_cases"] = a["similar_cases"]
        self.case.provenance["similar_cases"] = {"query": "case memory cosine kNN (closed cases opened before trigger time)",
                                                 "via": "case memory", "ids": [x["case_id"] for x in a["similar_cases"]]}
        self.case.log("tool", {"tool": "find_similar_cases", "n": len(a["similar_cases"])})
        return {"similar_cases": a["similar_cases"], "precedent_fraud_rate": a["precedent_fraud_rate"]}

    def search_policy(self, query, patterns=None):
        pack = self.kb.context_pack(query, patterns or [])
        flat = [dict(c, text=c["text"][:700]) for k in pack for c in pack[k]]
        self.policy_hits.extend(c for c in flat if c["chunk_id"] not in {h["chunk_id"] for h in self.policy_hits})
        self.case.provenance["policy"] = {"query": f"GraphRAG: {query}", "via": "policy knowledge base (PolicyChunk)",
                                          "ids": [h["chunk_id"] for h in self.policy_hits]}
        self.case.log("tool", {"tool": "search_policy", "query": query, "hits": [c["chunk_id"] for c in flat]})
        return pack

    # ---------------------------------------------------------------- engine tools
    def _assess(self, silent=False):
        self._need_tx()
        a = assess(self.bundle, self.mem, self.evidence_log, exclude_case_ids={self.case.case_id})
        if not silent:
            self.assessment = a
            self.assessed_since_evidence = True
        return a

    def assess_risk(self):
        a = self._assess()
        stage = "initial" if not self.evidence_log else f"after_evidence_{len(self.evidence_log)}"
        snap = {"stage": stage, "p_fraud": a["p_fraud"], "p_fraud_graph_only": a["p_fraud_graph_only"],
                "confidence": a["confidence"], "band": a["band"], "enough_evidence": a["enough_evidence"],
                "stop_reason": a["stop_reason"], "likely_fraud_type": a["likely_fraud_type"],
                "patterns": [{k: p[k] for k in ("name", "score", "documented")} for p in a["patterns"][:3]],
                "evidence_updates": a["evidence_updates"], "confidence_components": a["confidence_components"],
                "unresolved_questions": a["unresolved_questions"]}
        self.case.assessments.append(snap)
        self.case.unresolved_questions = a["unresolved_questions"]
        self._record_decision(stage, a)
        self.case.set_status("investigating" if not a["enough_evidence"] else "pending_approval", a["stop_reason"])
        self.case.log("assessment", snap)
        return {k: a[k] for k in ("p_fraud", "p_fraud_graph_only", "confidence", "band", "enough_evidence", "stop_reason", "unresolved_questions",
                                  "likely_fraud_type", "patterns", "top_contributions", "precedent_fraud_rate", "evidence_updates",
                                  "next_evidence_options", "recommended_evidence", "sar", "recommended_actions", "model",
                                  "confidence_components")}

    def _record_decision(self, stage, a):
        self.case.decisions = [d for d in self.case.decisions if d["stage"] != stage]
        self.case.decisions.append({"stage": stage, "at_evidence_round": len(self.evidence_log), "p_fraud": a["p_fraud"],
                                    "confidence": a["confidence"], "band": a["band"],
                                    "next_best_actions": a["recommended_actions"],
                                    "recommended_evidence": (a["recommended_evidence"] or {}).get("request"),
                                    "approval_routes": sorted({x["route"] for x in a["recommended_actions"]}, key=lambda r: r)})

    def request_evidence(self, request, justification=""):
        spec = rules()["evidence_requests"].get(request)
        if not spec:
            return {"denied": True, "reason": "unknown request type"}
        if not self.assessed_since_evidence or not self.assessment:
            self.assess_risk()
        a = self.assessment
        if a["enough_evidence"]:
            self.case.log("policy_block", {"request": request, "reason": "evidence already sufficient"})
            return {"denied": True, "reason": "assessment already decisive; additional customer contact not justified"}
        allowed = {o["request"] for o in a["next_evidence_options"]}
        if request not in allowed:
            self.case.log("policy_block", {"request": request, "reason": "not applicable / already used / rounds exhausted"})
            return {"denied": True, "reason": "not permitted for this trigger, already used, or evidence rounds exhausted",
                    "permitted": sorted(allowed)}
        auth = authorize(request, {"amount": self.bundle["txn"].get("amount")})
        stage = f"before_evidence_{len(self.evidence_log) + 1}"
        self._record_decision(stage, a)
        ACT.execute(request, self.case.case_id, {"justification": justification})
        self.case.set_status("awaiting_evidence", f"{request} sent")
        resp = self.ev.get(self.case.case_id, request, a["p_fraud"])
        opt = next(o for o in a["next_evidence_options"] if o["request"] == request)
        entry = {"request": request, "justification": justification, "route": auth["route"],
                 "p_fraud_before": a["p_fraud"], "voi": opt["p_decisive"], "expected_outcomes": opt["outcomes"]}
        if not resp:
            entry.update({"outcome": "no_response" if "no_response" in spec["outcomes"] else "inconclusive", "source": "none"})
        else:
            entry.update(resp)
        entry["lr"] = evidence_lr(request, entry["outcome"])
        self.evidence_log.append(entry)
        self.case.evidence_requests.append(entry)
        self.assessed_since_evidence = False
        self.case.set_status("investigating", f"{request} returned {entry['outcome']}")
        self.case.log("evidence_received", entry)
        return {"outcome": entry["outcome"], "source": entry.get("source"), "likelihood_ratio": round(entry["lr"], 2),
                "note": "call assess_risk to update the posterior and recommendation"}

    def record_finding(self, text, source, kind="hypothesis", refs=None):
        self.case.add_finding(text, source, kind, refs, actor="llm")
        return {"ok": True, "n_findings": len(self.case.findings)}

    def finalize_case(self, fraud_type, summary, explanation, actions, sar_narrative="", deviations=""):
        if not self.assessed_since_evidence:
            self.assess_risk()
        a = self.assessment
        engine = {x["action"]: x for x in a["recommended_actions"]}
        final_actions = []
        for name in actions or list(engine):
            if name in engine:
                final_actions.append(dict(engine[name]))
            else:
                auth = authorize(name, {"amount": self.bundle["txn"].get("amount"), "override_of_engine": True})
                if not auth["allowed"]:
                    continue
                final_actions.append({"priority": len(final_actions) + 1, "action": name, "rationale": f"agent addition: {deviations}",
                                      "route": auth["route"], "approver": auth["approver"], "execute_now": auth["execute_now"],
                                      "policy_note": auth["reason"]})
        dropped = [n for n in engine if n not in {x["action"] for x in final_actions}]
        for n in dropped:   # the LLM may not silently drop policy-mandated actions
            if n in ("file_sar", "escalate_to_analyst", "create_case"):
                final_actions.append(dict(engine[n], rationale=engine[n]["rationale"] + " (policy-mandated; retained)"))
        self.final = {"fraud_type": fraud_type, "summary": summary, "explanation": explanation, "actions": final_actions,
                      "sar_narrative": sar_narrative, "deviations": deviations, "dropped_engine_actions": dropped}
        return {"ok": True, "final_actions": [x["action"] for x in final_actions]}

    # ---------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict):
        fn = getattr(self, name, None)
        if not fn or name.startswith("_") or name not in {t["name"] for t in TOOLS}:
            return {"error": f"unknown tool {name}"}
        return fn(**(args or {}))


# ============================================================================== runners
def _compact(obj, limit=6000) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + '..."truncated"'


class Agent:
    def __init__(self, backend: GraphBackend | None = None, mem: CaseMemory | None = None, kb: KnowledgeBase | None = None,
                 evidence: EvidenceProvider | None = None, use_llm: bool | None = None, model: str | None = None):
        self.be = backend or get_backend()
        self.mem = mem or get_memory()
        self.kb = kb or get_kb()
        self.ev = evidence or make_provider()
        self.use_llm = settings.llm_enabled() if use_llm is None else use_llm
        self.model = model or settings.LLM_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def investigate(self, case_id: str, txn_id: str, trigger_type: str, trigger_detail: str = "",
                    customer_id: str = "", is_benchmark: bool = False, on_event=None) -> tuple[Case, Investigation]:
        case = Case(case_id=case_id, trigger_type=trigger_type or "risk_signal", trigger_detail=trigger_detail,
                    txn_id=str(txn_id), customer_id=customer_id, is_benchmark=is_benchmark)
        case.log("trigger", {"type": case.trigger_type, "detail": trigger_detail, "txn_id": case.txn_id}, actor="system")
        case.set_status("investigating", "trigger received")
        inv = Investigation(case, self.be, self.mem, self.kb, self.ev)
        inv.on_event = on_event
        t0 = time.time()
        mode = "deterministic"
        if self.use_llm:
            try:
                self._llm_loop(inv)
                mode = f"llm:{self.model}"
            except Exception as e:
                case.log("llm_error", {"error": str(e)[:500], "trace": traceback.format_exc()[-800:]}, actor="system")
        self._complete(inv)            # guarantee the mandatory steps + finalize
        self._close_out(inv, mode, time.time() - t0)
        return case, inv

    # ---------------------------------------------------------------- LLM loop
    def _llm_loop(self, inv: Investigation):
        c = inv.case
        user = (f"New investigation {c.case_id}.\nTrigger type: {c.trigger_type}\nTrigger detail: {c.trigger_detail or '(none)'}\n"
                f"Trigger transaction: {c.txn_id}\nCustomer (if known): {c.customer_id or 'unknown'}\n"
                f"Investigate, decide, and finalize the case.")
        messages = [{"role": "user", "content": user}]
        for turn in range(MAX_TURNS):
            resp = self.client.messages.create(
                model=self.model, max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS, messages=messages, thinking={"type": "adaptive"},
                output_config={"effort": settings.LLM_EFFORT})
            if resp.stop_reason == "refusal":
                c.log("llm_refusal", {"turn": turn}, actor="llm")
                return
            messages.append({"role": "assistant", "content": resp.content})
            texts = [b.text for b in resp.content if b.type == "text" and b.text.strip()]
            if texts:
                c.log("llm_reasoning", {"text": "\n".join(texts)[:3000]}, actor="llm")
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                if resp.stop_reason in ("end_turn", "stop_sequence") and inv.final is None:
                    messages.append({"role": "user", "content": "Please call finalize_case to complete the case."})
                    continue
                return
            results = []
            for u in uses:
                c.log("tool_call", {"tool": u.name, "input": u.input}, actor="llm")
                if getattr(inv, "on_event", None):
                    inv.on_event("tool_call", {"tool": u.name, "input": u.input})
                try:
                    out, err = inv.call(u.name, dict(u.input or {})), False
                except Exception as e:
                    out, err = {"error": str(e)[:400]}, True
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": _compact(out), "is_error": err})
            messages.append({"role": "user", "content": results})
            if inv.final is not None:
                return

    # ---------------------------------------------------------------- deterministic plan / completeness check
    def _complete(self, inv: Investigation):
        c = inv.case
        auto = []
        for key, fn in (("txn", inv.get_transaction), ("profile", inv.get_customer_profile), ("links", inv.get_shared_entity_links),
                        ("ring", inv.expand_fraud_ring), ("related_cases", inv.get_related_cases)):
            if key not in inv.bundle:
                fn(); auto.append(key)
        if not inv.bundle.get("txn"):
            return
        if "similar_cases" not in c.evidence:
            inv.find_similar_cases(); auto.append("similar_cases")
        if inv.final is None:
            if not inv.assessed_since_evidence:
                inv.assess_risk()
            # evidence loop (the LLM may already have done this)
            while not inv.assessment["enough_evidence"] and inv.assessment["recommended_evidence"]:
                best = inv.assessment["recommended_evidence"]
                inv.request_evidence(best["request"], f"engine: {best['p_decisive']:.0%} chance of decisive answer")
                inv.assess_risk()
                auto.append(best["request"])
            a = inv.assessment
            pats = [p["name"] for p in a["patterns"][:2]]
            if not inv.policy_hits:
                inv.search_policy(f"{c.trigger_type} {' '.join(pats)} actions approval SAR", pats)
            self._auto_findings(inv)
            inv.finalize_case(a["likely_fraud_type"], self._template_summary(inv), self._template_explanation(inv),
                              [x["action"] for x in a["recommended_actions"]])
        if auto:
            c.log("completeness_check", {"auto_ran": auto}, actor="system")

    def _auto_findings(self, inv: Investigation):
        a, tx, ev = inv.assessment, inv.bundle["txn"], inv.case.evidence
        tid, cid = tx["txn_id"], tx.get("customer_id")
        prof = ev.get("customer_profile") or {}
        add = inv.case.add_finding
        add(f"Transaction {tid}: ${float(tx['amount']):,.2f}, product {tx.get('product')}, bank model score "
            f"{float(tx.get('risk_score') or 0):.3f}", "get_transaction", "observed", [tid, cid])
        if prof:
            add(f"Customer {cid} had {prof.get('n_txn_180d')} transactions in the prior 180 days; trigger device known: "
                f"{prof.get('trigger_device_known')}, email known: {prof.get('trigger_email_known')}, card known: "
                f"{prof.get('trigger_card_known')}", "get_customer_profile", "observed", [cid, tx.get("device_key"), tx.get("p_email")])
        for l in ev.get("shared_entities") or []:
            if not l.get("hub") and l.get("other_customers"):
                add(f"{l['etype']} {l['eid']} is shared with {l['other_customers']} other customer(s), "
                    f"{l.get('fraud_customers', 0)} with confirmed-fraud history", "get_shared_entity_links", "observed",
                    [f"{l['etype']}:{l['eid']}"])
        for t in a["top_contributions"][:5]:
            lo = t["log_odds"] or 0
            if abs(lo) > 0.15:
                what = f"{t['meaning']} = {t['value']}" if t["value"] else f"Absent: {t['meaning'].lower()}"
                add(f"{what} ({lo:+.2f} log-odds toward {'fraud' if lo > 0 else 'legitimate'})",
                    "assess_risk (calibrated model)", "computed", [tid])
        if a["similar_cases"]:
            x = a["similar_cases"][0]
            add(f"Most similar prior case {x['case_id']} (similarity {x['score']}) was {x['outcome']}"
                f"{' as ' + x['fraud_type'] if x['fraud_type'] else ''}", "find_similar_cases", "computed", [x["case_id"]])
        if a["likely_fraud_type"] != "none (likely legitimate)":
            add(f"Likely pattern: {a['likely_fraud_type']} (model-generated hypothesis, not confirmed)", "patterns.identify",
                "hypothesis", [p["name"] for p in a["patterns"][:2]])

    def _template_summary(self, inv):
        a, tx, c = inv.assessment, inv.bundle["txn"], inv.case
        return (f"{c.trigger_type} investigation of transaction {tx['txn_id']} (${float(tx['amount']):,.2f}, product {tx.get('product')}, "
                f"model score {float(tx.get('risk_score') or 0):.2f}) for customer {tx.get('customer_id')}. "
                f"Assessment: {a['band']} (P(fraud)={a['p_fraud']:.2f}, confidence {a['confidence']:.2f}); likely type: {a['likely_fraud_type']}. "
                f"Additional evidence: {', '.join(e['request'] + '=' + e['outcome'] for e in inv.evidence_log) or 'none required'}. "
                f"{'SAR required. ' if a['sar']['required'] else ''}Decision basis: {a['stop_reason']}.")

    def _template_explanation(self, inv):
        a = inv.assessment
        fmt = lambda x: f"{x['meaning']} = {x['value']}" if x["value"] else f"absent: {x['meaning'].lower()}"
        pos = [fmt(x) for x in a["top_contributions"] if (x["log_odds"] or 0) > 0][:5]
        neg = [fmt(x) for x in a["top_contributions"] if (x["log_odds"] or 0) < 0][:4]
        req = "; ".join(f"{e['request']} was requested because P(fraud) was {e['p_fraud_before']:.2f} (uncertain) and it had a "
                        f"{e['voi']:.0%} chance of a decisive answer -> {e['outcome']} (LR {e['lr']:.2f})" for e in inv.evidence_log)
        acts = "; ".join(f"{x['action']} [{x['route']}]: {x['rationale']}" for x in a["recommended_actions"])
        pol = "; ".join(f"[{h['chunk_id']}] {h['source']} § {h['heading']}" for h in inv.policy_hits[:4])
        prec = ", ".join(f"{x['case_id']} ({x['outcome']}, sim {x['score']})" for x in a["similar_cases"][:3])
        return (f"Evidence toward fraud: {', '.join(pos) or 'none material'}. Evidence toward legitimate: {', '.join(neg) or 'none material'}. "
                f"Precedent: similar prior cases {prec or 'none'}; weighted fraud rate {a['precedent_fraud_rate']}. "
                f"Additional evidence: {req or 'not needed - initial assessment decisive' if a['enough_evidence'] and not inv.evidence_log else req or 'none available'}. "
                f"Actions: {acts}. Policy references: {pol or 'n/a'}. "
                f"Why the investigation stopped: {a['stop_reason']}. "
                f"Remaining uncertainty: confidence {a['confidence']:.2f}; open questions: {'; '.join(a['unresolved_questions']) or 'none'}.")

    # ---------------------------------------------------------------- execution, persistence, memory
    def _close_out(self, inv: Investigation, mode: str, secs: float):
        c, a, f = inv.case, inv.assessment, inv.final
        if not a or not f:
            c.set_status("escalated", "investigation could not complete (missing transaction?)")
            c.save()
            return
        c.fraud_type = f["fraud_type"]
        c.summary, c.narrative = f["summary"], f["explanation"]
        for act in f["actions"]:
            if act["execute_now"]:
                rec = ACT.execute(act["action"], c.case_id, {"rationale": act["rationale"]})
                c.actions_taken.append({**act, "execution": rec})
                c.log("action_executed", {"action": act["action"], "system": rec["system"], "id": rec["id"]})
            else:
                c.pending_approvals.append({**act, "status": "awaiting approval"})
                c.log("approval_requested", {"action": act["action"], "approver": act["approver"], "route": act["route"]})
        c.decisions.append({"stage": "final", "at_evidence_round": len(inv.evidence_log), "p_fraud": a["p_fraud"],
                            "confidence": a["confidence"], "band": a["band"], "next_best_actions": f["actions"],
                            "approval_routes": sorted({x["route"] for x in f["actions"]}),
                            "deviations_from_engine": f.get("deviations") or "", "dropped_engine_actions": f.get("dropped_engine_actions")})
        if a["sar"]["required"]:
            c.sar = SAR.draft(c.to_dict(), a, inv.bundle, f.get("sar_narrative") or None)
            c.log("sar_drafted", {"basis": a["sar"]["reasons"], "route": "compliance"})
        c.outcome = {"fraud": "suspected_fraud", "legitimate": "likely_legitimate"}.get(a["band"], "undetermined")
        c.set_status({"fraud": "pending_approval", "legitimate": "resolved_legitimate"}.get(a["band"], "escalated")
                     if c.pending_approvals or a["band"] != "legitimate" else "resolved_legitimate",
                     f"final decision: {a['band']}")
        c.log("closed_out", {"mode": mode, "seconds": round(secs, 1), "mcp_calls": len(getattr(self.be, "mcp_calls", []))}, actor="system")
        # write to graph + case memory
        try:
            self.be.write_case({"case_id": c.case_id, "status": c.status, "outcome": c.outcome, "fraud_type": c.fraud_type,
                                "trigger_type": c.trigger_type, "risk": a["p_fraud"], "confidence": a["confidence"],
                                "decision": ";".join(x["action"] for x in f["actions"]),
                                "approval_route": ";".join(sorted({x["route"] for x in f["actions"]})),
                                "sar_required": a["sar"]["required"], "summary": c.summary, "opened_ts": c.opened_ts,
                                "is_benchmark": c.is_benchmark, "txn_id": c.txn_id, "customer_id": c.customer_id,
                                "patterns": [p for p in a["patterns"][:3] if p["score"] >= 0.3],
                                "similar_cases": [s for s in a["similar_cases"][:3] if not s.get("agent_case")]},
                               signal_vector(a["signals"]))
            self.be.write_events(c.case_id, c.events)
            c.log("graph_write", {"backend": self.be.name, "vertices": ["FraudCase", "CaseEvent", "Pattern"],
                                  "edges": ["CASE_TXN", "CASE_CUSTOMER", "HAS_EVENT", "MATCHES_PATTERN", "SIMILAR_TO"]}, actor="system")
        except Exception as e:
            c.log("graph_write_error", {"error": str(e)[:400]}, actor="system")
        self.mem.remember({"case_id": c.case_id, "outcome": "agent_" + c.outcome, "fraud_type": c.fraud_type,
                           "actions_taken": [x["action"] for x in f["actions"]], "summary": c.summary, "opened_ts": c.opened_ts},
                          a["signals"])
        c.save()
