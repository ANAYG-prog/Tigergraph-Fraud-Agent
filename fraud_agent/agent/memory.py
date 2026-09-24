"""Case memory: prior investigations as vectors + outcomes, and the risk model learned from them.

build()   – for every closed case, gather point-in-time graph evidence, compute signals,
            store the signal vector on the FraudCase vertex (TigerGraph) and locally,
            and fit a calibrated logistic model P(fraud | signals) on analyst outcomes.
similar() – nearest prior cases (cosine over standardized signal vectors) with outcomes,
            fraud types and the analyst's actions: precedent for the recommendation.
remember()– add a newly resolved case so later investigations can use it.
"""
from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .. import settings
from ..data.dataset import load
from ..graph.backend import get_backend
from . import signals as S
from .investigate import gather

CACHE = settings.CACHE_DIR / f"case_memory_{settings.data_tag()}.pkl"

# Used only when too few labelled cases exist to fit a model (log-odds per unit of signal).
PRIOR_WEIGHTS = {
    "new_device": 0.7, "new_email": 0.5, "anon_email": 0.8, "proxy": 0.9, "small_probe_24h": 0.35, "burst_1h": 0.2,
    "shared_device_customers": 0.5, "linked_fraud_customers": 0.8, "ring_fraud_members": 0.6, "related_case_fraud": 0.7,
    "related_case_cleared": -0.6, "customer_prior_cleared": -0.4, "thin_history": 0.3, "amount_over_max": 0.4,
    "trigger_customer_report": 1.0, "new_card": 0.4, "email_mismatch": 0.3, "product_c": 0.3,
}


class CaseMemory:
    def __init__(self):
        self.cases: list[dict] = []          # {case_id, outcome, fraud_type, actions, summary, signals, vec}
        self.mu = np.zeros(len(S.SIGNAL_NAMES))
        self.sd = np.ones(len(S.SIGNAL_NAMES))
        self.model: LogisticRegression | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------ model
    def _z(self, v):
        return (np.asarray(v, float) - self.mu) / self.sd

    def fit(self):
        lab = [c for c in self.cases if c["outcome"] in ("fraud", "cleared")]
        if not lab:
            return
        X = np.array([c["vec"] for c in lab], float)
        y = np.array([c["outcome"] == "fraud" for c in lab], int)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        self.metrics = {"n_cases": len(lab), "fraud_rate": float(y.mean())}
        if len(lab) >= 30 and 0 < y.sum() < len(y):
            self.model = LogisticRegression(C=0.3, max_iter=2000)
            Z = self._z(X)
            try:
                cv = cross_val_predict(LogisticRegression(C=0.3, max_iter=2000), Z, y, cv=5, method="predict_proba")[:, 1]
                from sklearn.metrics import roc_auc_score, brier_score_loss
                base_auc = roc_auc_score(y, X[:, S.SIGNAL_NAMES.index("model_risk")])
                self.metrics.update({"cv_auc": round(float(roc_auc_score(y, cv)), 4),
                                     "model_risk_only_auc": round(float(base_auc), 4),
                                     "cv_brier": round(float(brier_score_loss(y, cv)), 4)})
            except Exception:
                pass
            self.model.fit(Z, y)

    def score(self, sig: dict) -> tuple[float, dict[str, float]]:
        """Posterior log-odds and per-signal contributions (log-odds units)."""
        v = np.array(S.vector(sig), float)
        if self.model is not None:
            z = self._z(v)
            contrib = dict(zip(S.SIGNAL_NAMES, (self.model.coef_[0] * z).tolist()))
            return float(self.model.intercept_[0] + sum(contrib.values())), contrib
        r = min(max(sig.get("model_risk", 0.05), 1e-3), 1 - 1e-3)
        contrib = {"model_risk": float(np.log(r / (1 - r)))}
        for k, w in PRIOR_WEIGHTS.items():
            if sig.get(k):
                contrib[k] = w * float(min(sig[k], 3))
        return sum(contrib.values()), contrib

    # ------------------------------------------------------------------ retrieval
    def similar(self, sig: dict, k: int = 5, exclude: set | None = None, before_ts: int | None = None) -> list[dict]:
        pool = [c for c in self.cases if c["case_id"] not in (exclude or set())
                and (before_ts is None or (c.get("opened_ts") or 0) < before_ts)]
        if not pool:
            return []
        M = self._z(np.array([c["vec"] for c in pool]))
        q = self._z(np.array(S.vector(sig)))
        if self.model is not None:          # weight dimensions by how much they matter for fraud
            w = np.abs(self.model.coef_[0]) + 0.05
            M, q = M * w, q * w
        s = (M @ q) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(q) or 1) + 1e-9)
        idx = np.argsort(-s)[:k]
        return [{"case_id": pool[i]["case_id"], "score": round(float(s[i]), 3), "outcome": pool[i]["outcome"],
                 "fraud_type": pool[i].get("fraud_type") or "", "actions": pool[i].get("actions") or "",
                 "summary": (pool[i].get("summary") or "")[:300], "agent_case": pool[i].get("agent_case", False)}
                for i in idx]

    def remember(self, case: dict, sig: dict, analyst_confirmed: bool = False):
        self.cases = [c for c in self.cases if c["case_id"] != case["case_id"]]
        self.cases.append({"case_id": case["case_id"], "outcome": case.get("outcome") or "pending",
                           "fraud_type": case.get("fraud_type") or "", "actions": ";".join(case.get("actions_taken", [])),
                           "summary": case.get("summary", ""), "signals": sig, "vec": S.vector(sig),
                           "opened_ts": case.get("opened_ts"), "agent_case": not analyst_confirmed})
        if analyst_confirmed and len([c for c in self.cases if c["outcome"] in ("fraud", "cleared")]) >= 30:
            self.fit()   # analyst-resolved outcomes feed back into the risk model
        self.save()

    def save(self):
        CACHE.write_bytes(pickle.dumps(self))


def build(max_cases: int | None = None, workers: int = 4, write_graph: bool = True, verbose: bool = True) -> CaseMemory:
    ds, be = load(), get_backend()
    mem = CaseMemory()
    rows = ds.closed_cases.to_dict("records")[: max_cases or None]
    t0 = time.time()

    def one(r):
        if not r["txn_id"]:
            return None
        b = gather(be, r["txn_id"], r["trigger"], r["customer_id"] or None)
        if not b.get("txn"):
            return None
        sig = S.compute(b)
        return {"case_id": r["case_id"], "outcome": r["outcome"], "fraud_type": r["fraud_type"], "actions": r["actions"],
                "summary": r["trigger_text"], "signals": sig, "vec": S.vector(sig), "opened_ts": int(b["txn"]["ts"]),
                "txn_id": r["txn_id"], "customer_id": b["txn"].get("customer_id")}

    with ThreadPoolExecutor(workers) as ex:
        for i, c in enumerate(ex.map(one, rows)):
            if c:
                mem.cases.append(c)
            if verbose and i % 100 == 0:
                print(f"  memory: {i}/{len(rows)} cases ({time.time() - t0:.0f}s)")
    mem.fit()
    if write_graph:
        for c in mem.cases:
            be.write_case({"case_id": c["case_id"], "status": "closed", "outcome": c["outcome"], "fraud_type": c["fraud_type"],
                           "decision": c["actions"], "summary": c["summary"], "txn_id": c["txn_id"],
                           "customer_id": c["customer_id"], "opened_ts": c["opened_ts"],
                           "risk": c["signals"]["model_risk"], "is_benchmark": False}, c["vec"])
    mem.save()
    if verbose:
        print("  memory built:", json.dumps(mem.metrics))
    return mem


@lru_cache
def get_memory() -> CaseMemory:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    return build(write_graph=settings.GRAPH_BACKEND == "local")
