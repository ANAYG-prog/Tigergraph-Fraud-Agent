"""Graph evidence gathering primitives (each maps to one or more GSQL queries via MCP)."""
from __future__ import annotations

import time

from ..graph.backend import GraphBackend


def gather(backend: GraphBackend, txn_id: str, trigger_type: str = "", customer_id: str | None = None,
           steps: tuple = ("txn", "profile", "links", "ring", "community", "related_cases")) -> dict:
    """Full point-in-time evidence bundle for one trigger transaction."""
    b: dict = {"trigger_type": trigger_type, "timings": {}, "errors": {}}

    def run(name, fn):
        t0 = time.perf_counter()
        try:
            b[name] = fn()
        except Exception as e:  # graph errors become evidence gaps, not crashes
            b[name] = None
            b["errors"][name] = str(e)[:300]
        b["timings"][name] = round(time.perf_counter() - t0, 3)

    run("txn", lambda: backend.txn(txn_id))
    if not b.get("txn"):
        return b
    tx = b["txn"]
    cust = tx.get("customer_id") or customer_id
    tx["customer_id"] = cust
    if "profile" in steps:
        run("profile", lambda: backend.profile(cust, int(tx["ts"])))
    if "links" in steps:
        run("links", lambda: backend.entity_links(txn_id))
    if "ring" in steps:
        run("ring", lambda: backend.ring(cust, int(tx["ts"])))
    if "community" in steps:
        run("community", lambda: backend.community(cust))
    if "related_cases" in steps:
        run("related_cases", lambda: backend.related_cases(txn_id))
    return b
