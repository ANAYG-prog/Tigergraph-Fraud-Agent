"""Graph backends exposing the investigation queries with one normalized interface.

TigerGraphBackend – every call is an MCP tool call to tigergraph-mcp (installed GSQL
                    queries, vertex/edge upserts, vector search). This is the production path.
LocalBackend      – in-process pandas/NetworkX mirror of the same GSQL semantics, for offline
                    development and unit tests only.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

from .. import settings
from ..data.dataset import Dataset, load

ENTITY_COLS = {"Device": "device_key", "Network": "net_key", "Card": "card_key",
               "Address": "addr_key", "Email": "p_email"}
LINK_TYPES = ("Device", "Network", "Card")      # used for ring / community expansion
MAX_HUB = int(os.getenv("MAX_HUB", "60"))
MAX_RING = int(os.getenv("MAX_RING", "500"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GraphBackend:
    name = "base"
    mcp_calls: list

    # read
    def txn(self, txn_id: str) -> dict: ...
    def profile(self, customer_id: str, before_ts: int) -> dict: ...
    def entity_links(self, txn_id: str) -> list[dict]: ...
    def ring(self, customer_id: str, before_ts: int, hops: int = 3) -> dict: ...
    def related_cases(self, txn_id: str) -> list[dict]: ...
    def recent(self, customer_id: str, before_ts: int, k: int = 25) -> list[dict]: ...
    def community(self, customer_id: str) -> dict: ...
    def case_vectors(self) -> list[dict]: ...
    def policy_chunks(self) -> list[dict]: ...
    # write
    def write_case(self, case: dict, vector: list[float] | None): ...
    def write_events(self, case_id: str, events: list[dict]): ...


# =============================================================================== local
class LocalBackend(GraphBackend):
    name = "local"

    def __init__(self, ds: Dataset | None = None):
        self.ds = ds or load()
        self.mcp_calls = []
        t = self.ds.txns
        self.t = t
        self.pos_by_cust = t.groupby("customer_id").indices
        # positions in the FULL frame (t has a RangeIndex, so group labels == positions)
        self.pos_by_ent = {et: {k: v.to_numpy() for k, v in t[t[c] != ""].groupby(c).groups.items()}
                           for et, c in ENTITY_COLS.items()}
        self.degree = {et: t[t[c] != ""].groupby(c)["customer_id"].nunique().to_dict() for et, c in ENTITY_COLS.items()}
        self.cases: dict[str, dict] = {}
        self.events: dict[str, list] = defaultdict(list)
        self.prior = defaultdict(lambda: {"fraud": 0, "cleared": 0})
        # (entity -> [(customer, first_ts)]) and (customer -> [(entity, first_ts)]) for fast point-in-time BFS
        self.ent_custs, self.cust_ents = {}, {}
        for et in LINK_TYPES:
            col = ENTITY_COLS[et]
            g = t[t[col] != ""].groupby([col, "customer_id"], sort=False)["ts"].min().reset_index()
            ec, ce = defaultdict(list), defaultdict(list)
            for k, c, ts in g.itertuples(index=False):
                ec[k].append((c, ts)); ce[c].append((k, ts))
            self.ent_custs[et], self.cust_ents[et] = ec, ce
        self._seed_cases()
        self._communities()

    def _seed_cases(self):
        for r in self.ds.closed_cases.to_dict("records"):
            tx = self.ds.txn(r["txn_id"]) or {}
            cust = r["customer_id"] or tx.get("customer_id", "")
            self.cases[r["case_id"]] = {"case_id": r["case_id"], "outcome": r["outcome"] or "", "fraud_type": r["fraud_type"],
                                        "decision": r["actions"], "risk": float(tx.get("risk_score") or 0),
                                        "summary": r["trigger_text"], "txn_id": r["txn_id"], "customer_id": cust,
                                        "opened_ts": int(tx.get("ts") or 0), "is_benchmark": False}
            if cust and r["outcome"] in ("fraud", "cleared"):
                self.prior[cust][r["outcome"]] += 1

    def _communities(self):
        g = nx.Graph()
        g.add_nodes_from(self.pos_by_cust.keys())
        for et in LINK_TYPES:
            for key, pos in self.pos_by_ent[et].items():
                if 1 < self.degree[et].get(key, 0) <= MAX_HUB:
                    custs = self.t["customer_id"].values[pos]
                    u = list(dict.fromkeys(custs))
                    g.add_edges_from(zip(u, u[1:]))
        self.comm, self.comm_members = {}, {}
        for i, comp in enumerate(nx.connected_components(g)):
            for c in comp:
                self.comm[c] = i
            self.comm_members[i] = comp

    # ------------------------------------------------------------------ reads
    def txn(self, txn_id):
        r = self.ds.txn(txn_id)
        if r is None:
            raise KeyError(f"transaction {txn_id} not found")
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}

    def _hist(self, customer_id, before_ts, window_days=180):
        pos = self.pos_by_cust.get(customer_id, [])
        h = self.t.iloc[pos]
        return h[(h.ts < before_ts) & (h.ts >= before_ts - window_days * 86400)]

    def profile(self, customer_id, before_ts):
        h = self._hist(customer_id, before_ts)
        last24 = h[h.ts >= before_ts - 86400]
        return {"n_txn": len(h), "sum_amt": float(h.amount.sum()), "sum_amt2": float((h.amount ** 2).sum()),
                "max_amt": float(h.amount.max()) if len(h) else 0.0, "max_risk": float(h.risk_score.max()) if len(h) else 0.0,
                "first_ts": int(h.ts.min()) if len(h) else None, "last_ts": int(h.ts.max()) if len(h) else None,
                "n_7d": int((h.ts >= before_ts - 7 * 86400).sum()), "n_24h": len(last24),
                "n_1h": int((h.ts >= before_ts - 3600).sum()), "small_24h": int((last24.amount < 10).sum()),
                "amt_24h": float(last24.amount.sum()),
                "devices": sorted(set(h.device_key) - {""}), "emails": sorted(set(h.p_email) - {""}),
                "cards": sorted(set(h.card_key) - {""}), "addrs": sorted(set(h.addr_key) - {""}),
                "nets": sorted(set(h.net_key) - {""})}

    def entity_links(self, txn_id):
        tx = self.txn(txn_id)
        out = []
        for et, col in ENTITY_COLS.items():
            key = tx.get(col) or ""
            if not key:
                continue
            deg = self.degree[et].get(key, 0)
            if deg > MAX_HUB:
                out.append({"etype": et, "eid": key, "degree": deg, "hub": True})
                continue
            rows = self.t.iloc[self.pos_by_ent[et][key]]
            rows = rows[(rows.ts <= tx["ts"]) & (rows.txn_id != tx["txn_id"]) & (rows.customer_id != tx["customer_id"])]
            custs = sorted(set(rows.customer_id))
            out.append({"etype": et, "eid": key, "degree": deg, "hub": False, "other_customers": len(custs),
                        "other_txns": len(rows), "avg_risk": float(rows.risk_score.mean()) if len(rows) else 0.0,
                        "other_amount": float(rows.amount.sum()),
                        "fraud_customers": sum(1 for c in custs if self.prior[c]["fraud"] > 0),
                        "cleared_customers": sum(1 for c in custs if self.prior[c]["cleared"] > 0),
                        "customer_ids": custs[:25]})
        return out

    def ring(self, customer_id, before_ts, hops=3, max_members=MAX_RING):
        seen, depth, frontier, ents = {customer_id}, {customer_id: 0}, {customer_id}, set()
        for h in range(1, hops + 1):
            nxt = set()
            for c in frontier:
                for et in LINK_TYPES:
                    for key, ts in self.cust_ents[et].get(c, []):
                        if ts > before_ts or not (1 < self.degree[et].get(key, 0) <= MAX_HUB):
                            continue
                        ents.add((et, key))
                        for m, ts2 in self.ent_custs[et][key]:
                            if ts2 <= before_ts and m not in seen:
                                seen.add(m); depth[m] = h; nxt.add(m)
            frontier = nxt
            if not frontier or len(seen) >= max_members:
                break
        pos = np.concatenate([self.pos_by_cust[m] for m in seen]) if seen else []
        r = self.t.iloc[pos]
        r = r[(r.ts <= before_ts) & (r.ts >= before_ts - 30 * 86400)]
        members = [{"customer_id": m, "depth": depth[m], "prior_fraud": self.prior[m]["fraud"],
                    "prior_cleared": self.prior[m]["cleared"], "community": self.comm.get(m, -1)} for m in seen]
        return {"ring_size": len(seen), "shared_entities": len(ents), "amount_30d": float(r.amount.sum()),
                "txns_30d": len(r), "avg_risk_30d": float(r.risk_score.mean()) if len(r) else 0.0,
                "members": sorted(members, key=lambda x: x["depth"])[:60]}

    def related_cases(self, txn_id):
        tx = self.txn(txn_id)
        hits = {}
        for c in self.cases.values():
            if c["is_benchmark"] or c["opened_ts"] >= tx["ts"]:
                continue
            via = []
            if c["customer_id"] == tx["customer_id"]:
                via.append("same_customer")
            ctx = self.ds.txn(c["txn_id"]) if c.get("txn_id") else None
            if ctx and ctx["txn_id"] != tx["txn_id"]:
                for et in ("Device", "Network", "Card", "Address"):
                    col = ENTITY_COLS[et]
                    if tx.get(col) and ctx.get(col) == tx.get(col) and self.degree[et].get(tx[col], 0) <= MAX_HUB:
                        via.append(f"shared_{et}")
            if via:
                hits[c["case_id"]] = {k: c[k] for k in ("case_id", "outcome", "fraud_type", "decision", "risk", "summary")} | {"via": via}
        return list(hits.values())

    def recent(self, customer_id, before_ts, k=25):
        h = self.t.iloc[self.pos_by_cust.get(customer_id, [])]
        h = h[h.ts <= before_ts].tail(k)
        cols = ["txn_id", "ts", "amount", "product", "risk_score", "device_key", "p_email", "card_key", "proxy"]
        return h[cols].iloc[::-1].to_dict("records")

    def community(self, customer_id):
        cc = self.comm.get(customer_id, -1)
        mem = self.comm_members.get(cc, set())
        return {"community": cc, "size": len(mem), "fraud_members": sum(1 for m in mem if self.prior[m]["fraud"]),
                "cleared_members": sum(1 for m in mem if self.prior[m]["cleared"])}

    def case_vectors(self):
        return [c for c in self.cases.values() if c.get("embedding")]

    def policy_chunks(self):
        return getattr(self, "_chunks", [])

    def upsert_policy_chunks(self, chunks):
        self._chunks = chunks

    # ------------------------------------------------------------------ writes
    def write_case(self, case, vector):
        rec = self.cases.setdefault(case["case_id"], {})
        rec.update({k: case.get(k) for k in ("case_id", "outcome", "fraud_type", "decision", "risk", "summary",
                                              "txn_id", "customer_id", "opened_ts", "is_benchmark", "status")})
        if vector is not None:
            rec["embedding"] = list(vector)
        return {"ok": True}

    def write_events(self, case_id, events):
        self.events[case_id] = list(events)
        return {"ok": True}


# =============================================================================== TigerGraph (MCP)
class TigerGraphBackend(GraphBackend):
    name = "tigergraph"

    def __init__(self):
        from .mcp_client import TigerGraphMCP
        self.mcp = TigerGraphMCP()
        self.mcp_calls = self.mcp.calls
        self.graph = settings.TG_GRAPH

    def _q(self, name, **params):
        return self.mcp.run_query(name, params)

    @staticmethod
    def _find(results, key):
        for blk in results:
            if isinstance(blk, dict) and key in blk:
                return blk[key]
        return None

    def txn(self, txn_id):
        res = self._q("txn_context", t=str(txn_id))
        txv = (self._find(res, "txn") or [None])[0]
        if not txv:
            raise KeyError(f"transaction {txn_id} not found")
        out = dict(txv["attributes"])
        out["txn_id"] = txv["v_id"]
        for e in self._find(res, "entities") or []:
            rel = set(e["attributes"].get("@rel", []))
            vt, vid = e["v_type"], e["v_id"]
            if vt == "Customer":
                out["customer_id"] = vid
            elif vt == "Email":
                if "PURCHASER_EMAIL" in rel:
                    out["p_email"] = vid
                if "RECIPIENT_EMAIL" in rel:
                    out["r_email"] = vid
            elif vt in ENTITY_COLS:
                out[ENTITY_COLS[vt]] = vid
        for c in ENTITY_COLS.values():
            out.setdefault(c, "")
        out.setdefault("r_email", "")
        return out

    def profile(self, customer_id, before_ts):
        res = self._q("customer_profile", c=customer_id, before_ts=int(before_ts))
        p = {}
        for blk in res:
            p.update(blk)
        if p.get("first_ts") == 2147483647:
            p["first_ts"] = None
        return p

    def entity_links(self, txn_id):
        res = self._q("entity_links", t=str(txn_id), max_hub=MAX_HUB)
        ents = self._find(res, "Ents") or []
        out = []
        for e in ents:
            a = e["attributes"]
            out.append({"etype": e["v_type"], "eid": e["v_id"], "degree": a.get("degree", 0), "hub": False,
                        "other_customers": a.get("other_customers", 0), "other_txns": a.get("other_txns", 0),
                        "avg_risk": a.get("avg_risk", 0.0), "other_amount": a.get("other_amount", 0.0),
                        "fraud_customers": a.get("fraud_customers", 0), "cleared_customers": a.get("cleared_customers", 0),
                        "customer_ids": list(a.get("customer_ids", []))[:25]})
        return out

    def ring(self, customer_id, before_ts, hops=3):
        res = self._q("ring_expand", c=customer_id, before_ts=int(before_ts), hops=hops, max_hub=MAX_HUB, max_members=MAX_RING)
        summary = {}
        for blk in res:
            if "ring_size" in blk:
                summary = dict(blk)
        members = [{"customer_id": m["v_id"], "depth": m["attributes"].get("depth", 0),
                    "prior_fraud": m["attributes"].get("prior_fraud", 0),
                    "prior_cleared": m["attributes"].get("prior_cleared", 0),
                    "community": m["attributes"].get("community", -1)} for m in (self._find(res, "Members") or [])]
        summary["members"] = sorted(members, key=lambda x: x["depth"])[:60]
        return summary

    def related_cases(self, txn_id):
        res = self._q("related_cases", t=str(txn_id), max_hub=MAX_HUB)
        return [{"case_id": c["v_id"], **{k: c["attributes"].get(k) for k in ("outcome", "fraud_type", "decision", "risk", "summary")},
                 "via": list(c["attributes"].get("via", []))} for c in (self._find(res, "C") or [])]

    def recent(self, customer_id, before_ts, k=25):
        res = self._q("recent_txns", c=customer_id, before_ts=int(before_ts), k=k)
        return [{"txn_id": t["v_id"], **t["attributes"]} for t in (self._find(res, "T") or [])]

    def community(self, customer_id):
        res = self._q("community_stats", c=customer_id)
        out = {}
        for blk in res:
            out.update(blk)
        return out

    def case_vectors(self):
        p = self.mcp.call("tigergraph__get_nodes", {"graph_name": self.graph, "vertex_type": "FraudCase", "limit": 5000})
        rows = p.get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("vertices") or rows.get("results") or []
        out = []
        for v in rows:
            a = v.get("attributes", v)
            if a.get("embedding"):
                out.append({"case_id": v.get("v_id", a.get("case_id")), **a})
        return out

    def policy_chunks(self):
        p = self.mcp.call("tigergraph__get_nodes", {"graph_name": self.graph, "vertex_type": "PolicyChunk", "limit": 5000})
        rows = p.get("data") or []
        if isinstance(rows, dict):
            rows = rows.get("vertices") or rows.get("results") or []
        return [{"chunk_id": v.get("v_id"), **v.get("attributes", v)} for v in rows]

    def write_case(self, case, vector):
        attrs = {"case_id": case["case_id"], "status": case.get("status", ""), "outcome": case.get("outcome") or "",
                 "fraud_type": case.get("fraud_type") or "", "trigger_type": case.get("trigger_type", ""),
                 "risk": float(case.get("risk") or 0), "confidence": float(case.get("confidence") or 0),
                 "decision": case.get("decision") or "", "approval_route": case.get("approval_route") or "",
                 "sar_required": bool(case.get("sar_required")), "summary": (case.get("summary") or "")[:4000],
                 "opened_ts": int(case.get("opened_ts") or 0), "updated_at": _now(),
                 "is_benchmark": bool(case.get("is_benchmark"))}
        if vector is not None:
            attrs["embedding"] = [round(float(x), 5) for x in vector]
        self.mcp.call("tigergraph__add_nodes", {"graph_name": self.graph, "vertex_type": "FraudCase",
                                                "vertex_id": "case_id", "vertices": [attrs]})
        cid = case["case_id"]
        if case.get("txn_id"):
            self._edges("CASE_TXN", "FraudCase", "Transaction", [{"source_id": cid, "target_id": str(case["txn_id"]), "role": "trigger"}])
        if case.get("customer_id"):
            self._edges("CASE_CUSTOMER", "FraudCase", "Customer", [{"source_id": cid, "target_id": case["customer_id"]}])
        pats = case.get("patterns", []) or []
        if pats:
            self.mcp.call("tigergraph__add_nodes", {"graph_name": self.graph, "vertex_type": "Pattern", "vertex_id": "name",
                          "vertices": [{"name": p["name"], "description": p.get("description", ""),
                                        "documented": bool(p.get("documented", True))} for p in pats]})
            self._edges("MATCHES_PATTERN", "FraudCase", "Pattern",
                        [{"source_id": cid, "target_id": p["name"], "confidence": float(p.get("score", 0))} for p in pats])
        sims = case.get("similar_cases", []) or []
        if sims:
            self._edges("SIMILAR_TO", "FraudCase", "FraudCase",
                        [{"source_id": cid, "target_id": x["case_id"], "score": float(x.get("score", 0))} for x in sims])
        return {"ok": True}

    def _edges(self, etype, stype, ttype, edges):
        for e in edges:
            e["source_type"], e["target_type"] = stype, ttype
        return self.mcp.call("tigergraph__add_edges", {"graph_name": self.graph, "edge_type": etype, "edges": edges})

    def write_events(self, case_id, events):
        verts = [{"event_id": f"{case_id}#{e['seq']:03d}", "seq": e["seq"], "kind": e["kind"],
                  "detail": json.dumps(e.get("detail", {}), default=str)[:4000], "at": e.get("at", _now())} for e in events]
        if not verts:
            return {"ok": True}
        self.mcp.call("tigergraph__add_nodes", {"graph_name": self.graph, "vertex_type": "CaseEvent",
                                                "vertex_id": "event_id", "vertices": verts})
        self._edges("HAS_EVENT", "FraudCase", "CaseEvent",
                    [{"source_id": case_id, "target_id": v["event_id"]} for v in verts])
        return {"ok": True}


@lru_cache
def get_backend() -> GraphBackend:
    if settings.GRAPH_BACKEND == "local":
        return LocalBackend()
    return TigerGraphBackend()
