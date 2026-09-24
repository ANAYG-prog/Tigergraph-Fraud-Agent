"""Create the FraudGraph schema, bulk-load the dataset, install queries, run WCC, build case memory.

Bulk loading uses pyTigerGraph (REST upserts in batches) because pushing ~590k transactions
through an LLM tool protocol makes no sense; everything the *agent* does at runtime goes
through TigerGraph MCP.

    python scripts/setup_graph.py --all            # schema + load + queries + communities + memory + docs
    python scripts/setup_graph.py --load --queries # individual steps
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraud_agent import settings  # noqa: E402
from fraud_agent.data.dataset import load  # noqa: E402

G = settings.TG_GRAPH


def conn():
    import pyTigerGraph as tg
    host = os.environ["TG_HOST"]
    kw = dict(host=host, graphname=G, username=os.getenv("TG_USERNAME", "tigergraph"),
              password=os.getenv("TG_PASSWORD", "tigergraph"))
    if "tgcloud.io" in host:
        kw.update(restppPort=os.getenv("TG_RESTPP_PORT", "443"), gsPort=os.getenv("TG_GS_PORT", "443"))
    else:
        kw.update(restppPort=os.getenv("TG_RESTPP_PORT", "9000"), gsPort=os.getenv("TG_GS_PORT", "14240"))
    c = tg.TigerGraphConnection(**kw)
    secret = os.getenv("TG_SECRET")
    if secret:
        c.getToken(secret)
    return c


def schema(c):
    txt = (ROOT / "gsql" / "schema.gsql").read_text(encoding="utf-8")
    print(c.gsql(txt))
    # native vector attribute for case memory (TigerGraph >= 4.2); falls back to LIST<DOUBLE> embedding
    from fraud_agent.agent.signals import SIGNAL_NAMES
    try:
        print(c.gsql(f"""USE GRAPH {G}
CREATE GLOBAL SCHEMA_CHANGE JOB add_vec {{
  ALTER VERTEX FraudCase ADD VECTOR ATTRIBUTE case_vec(DIMENSION={len(SIGNAL_NAMES)}, METRIC="COSINE");
  ALTER VERTEX PolicyChunk ADD VECTOR ATTRIBUTE chunk_vec(DIMENSION=256, METRIC="COSINE");
}}
RUN GLOBAL SCHEMA_CHANGE JOB add_vec"""))
    except Exception as e:
        print("vector attributes not added (older TigerGraph?)", e)


def _upsert_df(c, df, vtype, pk, attrs, batch=5000):
    df = df.drop_duplicates(pk)
    for i in range(0, len(df), batch):
        c.upsertVertexDataFrame(df.iloc[i:i + batch], vtype, v_id=pk, attributes={a: a for a in attrs})
    print(f"  {vtype}: {len(df):,}")


def _upsert_edges(c, df, etype, stype, src, ttype, dst, batch=10000):
    df = df[(df[src] != "") & (df[dst] != "")].drop_duplicates([src, dst])
    for i in range(0, len(df), batch):
        c.upsertEdgeDataFrame(df.iloc[i:i + batch], stype, etype, ttype, from_id=src, to_id=dst, attributes={})
    print(f"  {etype}: {len(df):,}")


def load_data(c):
    ds = load()
    t = ds.txns.copy()
    t["risk_score"] = t["risk_score"].fillna(0.0)
    t["dist1"] = pd.to_numeric(t["dist1"], errors="coerce").fillna(-1.0)
    for col in ("proxy", "device_type", "product"):
        t[col] = t[col].fillna("").astype(str)
    t0 = time.time()
    _upsert_df(c, t, "Transaction", "txn_id", ["ts", "day", "amount", "product", "risk_score", "proxy", "device_type", "dist1"])

    prior = ds.closed_cases.merge(t[["txn_id", "customer_id"]].rename(columns={"customer_id": "cust_from_txn"}), on="txn_id", how="left")
    prior["cust"] = prior["customer_id"].where(prior["customer_id"] != "", prior["cust_from_txn"])
    pf = prior[prior.outcome == "fraud"].groupby("cust").size()
    pc = prior[prior.outcome == "cleared"].groupby("cust").size()
    cust = t.groupby("customer_id").agg(first_day=("day", "min"), n_txn=("txn_id", "size")).reset_index()
    cust["prior_fraud"] = cust["customer_id"].map(pf).fillna(0).astype(int)
    cust["prior_cleared"] = cust["customer_id"].map(pc).fillna(0).astype(int)
    cust["community"] = -1
    _upsert_df(c, cust, "Customer", "customer_id", ["first_day", "n_txn", "prior_fraud", "prior_cleared", "community"])

    for vtype, col, extra in (("Card", "card_key", {"brand": "card_brand", "card_type": "card_type"}),
                              ("Device", "device_key", {"device_type": "device_type"}),
                              ("Network", "net_key", {}), ("Email", "p_email", {}), ("Address", "addr_key", {})):
        sub = t[t[col] != ""]
        deg = sub.groupby(col)["customer_id"].nunique().rename("degree").reset_index()
        firsts = sub.drop_duplicates(col)[[col] + list(extra.values())].rename(columns={v: k for k, v in extra.items()})
        v = deg.merge(firsts, on=col)
        if vtype == "Email":   # include recipient-only domains
            r = t[(t.r_email != "") & (~t.r_email.isin(v[col]))][["r_email"]].drop_duplicates().rename(columns={"r_email": col})
            r["degree"] = 0
            v = pd.concat([v, r])
        _upsert_df(c, v, vtype, col, ["degree"] + list(extra))

    _upsert_edges(c, t, "MADE", "Customer", "customer_id", "Transaction", "txn_id")
    _upsert_edges(c, t, "PAID_WITH", "Transaction", "txn_id", "Card", "card_key")
    _upsert_edges(c, t, "USED_DEVICE", "Transaction", "txn_id", "Device", "device_key")
    _upsert_edges(c, t, "FROM_NETWORK", "Transaction", "txn_id", "Network", "net_key")
    _upsert_edges(c, t, "PURCHASER_EMAIL", "Transaction", "txn_id", "Email", "p_email")
    _upsert_edges(c, t, "RECIPIENT_EMAIL", "Transaction", "txn_id", "Email", "r_email")
    _upsert_edges(c, t, "BILLED_TO", "Transaction", "txn_id", "Address", "addr_key")
    print(f"loaded in {time.time() - t0:.0f}s")


def queries(c):
    txt = (ROOT / "gsql" / "queries.gsql").read_text(encoding="utf-8")
    print(c.gsql(txt))
    print(c.gsql(f"USE GRAPH {G}\nINSTALL QUERY ALL"))


def communities(c):
    print(c.runInstalledQuery("build_communities", {"max_hub": int(os.getenv("MAX_HUB", "60"))}, timeout=3_600_000))


def memory(c):
    """Closed cases -> FraudCase vertices (+vector), CASE_TXN/CASE_CUSTOMER edges; fit the risk model."""
    from fraud_agent.agent.memory import build
    mem = build(write_graph=False)
    rows = pd.DataFrame([{"case_id": m["case_id"], "status": "closed", "outcome": m["outcome"] or "",
                          "fraud_type": m["fraud_type"] or "", "decision": m["actions"] or "", "summary": (m["summary"] or "")[:2000],
                          "risk": float(m["signals"]["model_risk"]), "opened_ts": int(m["opened_ts"]), "is_benchmark": False,
                          "embedding": [round(float(x), 5) for x in m["vec"]], "txn_id": m["txn_id"],
                          "customer_id": m["customer_id"]} for m in mem.cases])
    _upsert_df(c, rows, "FraudCase", "case_id",
               ["status", "outcome", "fraud_type", "decision", "summary", "risk", "opened_ts", "is_benchmark", "embedding"])
    _upsert_edges(c, rows, "CASE_TXN", "FraudCase", "case_id", "Transaction", "txn_id")
    _upsert_edges(c, rows, "CASE_CUSTOMER", "FraudCase", "case_id", "Customer", "customer_id")
    try:
        c.upsertVertices("FraudCase", [(r.case_id, {"case_vec": list(map(float, r.embedding))}) for r in rows.itertuples()])
    except Exception as e:
        print("  native vectors skipped:", str(e)[:120])
    print("  model:", mem.metrics)


def docs(c):
    from fraud_agent.rag.knowledge import get_kb
    kb = get_kb()
    ch = pd.DataFrame([{"chunk_id": x.chunk_id, "source": x.source, "heading": x.heading, "text": x.text,
                        "embedding": [round(float(v), 5) for v in kb.mat[i]]} for i, x in enumerate(kb.chunks)])
    if ch.empty:
        print("  no policy documents found"); return
    _upsert_df(c, ch, "PolicyChunk", "chunk_id", ["source", "heading", "text", "embedding"])
    pats = pd.DataFrame([{"name": n, "description": p["description"][:2000], "documented": True} for n, p in kb.patterns.items()])
    if not pats.empty:
        _upsert_df(c, pats, "Pattern", "name", ["description", "documented"])
        e = pd.DataFrame([{"name": n, "chunk_id": cid} for n, p in kb.patterns.items() for cid in p["chunks"]])
        _upsert_edges(c, e, "DESCRIBED_BY", "Pattern", "name", "PolicyChunk", "chunk_id")
    print(f"  patterns documented: {list(kb.patterns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for s in ("schema", "load", "queries", "communities", "memory", "docs", "all"):
        ap.add_argument(f"--{s}", action="store_true")
    a = ap.parse_args()
    c = conn()
    steps = ["schema", "load", "queries", "communities", "memory", "docs"]
    for s in steps:
        if a.all or getattr(a, s):
            print(f"== {s}")
            globals()[{"load": "load_data"}.get(s, s)](c)
