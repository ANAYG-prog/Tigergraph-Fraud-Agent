"""Investigate the benchmark cases and write one answer file per case.

    python scripts/run_benchmark.py                 # all benchmark cases
    python scripts/run_benchmark.py --cases BM01 BM07 --no-llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fraud_agent import report, settings  # noqa: E402
from fraud_agent.agent.evidence import make_provider  # noqa: E402
from fraud_agent.agent.orchestrator import Agent  # noqa: E402
from fraud_agent.data.dataset import load  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--evidence", default=None, help="dataset | simulated")
    a = ap.parse_args()

    ds = load()
    bench = ds.benchmark
    if a.cases:
        bench = bench[bench.case_id.isin(a.cases)]
    agent = Agent(evidence=make_provider(a.evidence, ds.evidence), use_llm=False if a.no_llm else None)
    print(f"DATASET: {settings.DATASET_LABEL}")
    print(f"OUTPUT:  {settings.OUTPUT_DIR}")
    if not ds.readme or settings.IS_SYNTHETIC:
        print("WARNING: dataset README not read / synthetic data - answer format is provisional; not benchmark results.")
    print(f"{len(bench)} cases | backend={agent.be.name} | llm={'on ' + agent.model if agent.use_llm else 'off'} | evidence={agent.ev.mode}")
    rows = []
    for r in bench.to_dict("records"):
        t0 = time.time()
        case, inv = agent.investigate(r["case_id"], r["txn_id"], r["trigger"] or "risk_signal", r["trigger_text"],
                                      r["customer_id"], is_benchmark=True)
        path = report.write(case.to_dict())
        ans = json.loads(path.read_text())
        nba = ans["next_best_action"]
        rows.append({"case_id": case.case_id, "trigger": case.trigger_type, "outcome": case.outcome, "fraud_type": case.fraud_type,
                     "p_initial": case.decisions[0]["p_fraud"] if case.decisions else None,
                     "p_final": ans["determination"]["p_fraud"], "confidence": ans["determination"]["confidence"],
                     "evidence": "; ".join(f"{e['request']}={e['outcome']}" for e in ans["next_best_action"]["additional_evidence_requested"]),
                     "nba_before": (nba["before_additional_evidence"] or {}).get("next_best_action"),
                     "nba_final": (nba["final"] or {}).get("next_best_action"),
                     "sar": ans["sar_required"], "status": case.status, "secs": round(time.time() - t0, 1)})
        print(" ", rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(settings.OUTPUT_DIR / "benchmark_summary.csv", index=False)
    print(df.to_string(index=False))
