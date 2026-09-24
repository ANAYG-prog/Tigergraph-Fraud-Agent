import sys, types, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from fraud_agent.agent.orchestrator import Agent
from fraud_agent.data.dataset import load
from fraud_agent import report

def block(**kw): return types.SimpleNamespace(**kw)
script = [
  [("get_transaction", {}), ("get_customer_profile", {}), ("get_shared_entity_links", {})],
  [("expand_fraud_ring", {"hops": 2}), ("get_related_cases", {}), ("find_similar_cases", {})],
  [("assess_risk", {})],
  [("request_evidence", {"request": "request_customer_validation", "justification": "uncertain"})],
  [("assess_risk", {}), ("search_policy", {"query": "hold transaction customer validation", "patterns": ["account_takeover"]})],
  [("record_finding", {"text": "Customer confirmed the purchase", "source": "request_evidence"})],
  [("finalize_case", {"fraud_type": "none (legitimate)", "summary": "LLM summary", "explanation": "LLM explanation",
                      "actions": ["create_case", "allow_transaction", "close_case", "block_card"], "deviations": "test override"})],
]
class FakeMsgs:
    i = 0
    def create(self, **kw):
        assert kw["tools"] and kw["system"][0]["cache_control"]
        step = script[FakeMsgs.i]; FakeMsgs.i += 1
        content = [block(type="text", text=f"step {FakeMsgs.i}")] + [block(type="tool_use", id=f"t{FakeMsgs.i}_{j}", name=n, input=a) for j, (n, a) in enumerate(step)]
        return block(stop_reason="tool_use", content=content)
ds = load()
r = ds.benchmark.iloc[0].to_dict()
ag = Agent(use_llm=True); ag._client = block(messages=FakeMsgs())
c, inv = ag.investigate("MOCK1", r["txn_id"], r["trigger"], r["trigger_text"])
print("status", c.status, "| mode", [e for e in c.events if e["kind"]=="closed_out"][0]["detail"])
print("final actions", [(a["action"], a["route"]) for a in inv.final["actions"]])
print("stages", [d["stage"] for d in c.decisions]); print("errors", [e for e in c.events if "error" in e["kind"]])
