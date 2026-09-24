# Demo script (3–5 min)

1. **0:00 Problem (20s).** Analysts spend most of their time gathering evidence, and the money is often gone by the
   time a case closes.
2. **0:20 Architecture (30s).** Show the README diagram. Say: TigerGraph through MCP, GraphRAG, case memory, a decision
   engine, a policy gate and an LLM.
3. **0:50 Clear-cut fraud case (60s).** Pick a benchmark case the agent resolves as fraud without extra evidence.
   - Show the tool calls streaming in the status panel (GSQL queries through MCP).
   - Open the **Graph** tab: shared device → other customers → a red diamond for linked confirmed fraud.
   - Open **Decision & approvals**: `block_transaction` runs automatically, `block_card` waits for an analyst,
     `file_sar` waits for compliance.
4. **1:50 Uncertain case (70s).** Pick a case whose initial P(fraud) is around 0.4–0.6.
   - Show the `before_evidence_1` NBA: hold the transaction and request customer validation, with an 88% chance of a
     decisive answer.
   - Switch the sidebar to **scripted** evidence, set "customer denies", and run again. P(fraud) jumps and the NBA
     changes to block + card block + warn.
   - Run it once more with "customer confirms". The case clears.
5. **3:00 Customer report (30s).** Pick a report where the device, email and history match the customer. The agent
   does **not** ask the customer again; it types the case as friendly fraud and closes it with watchlisting.
6. **3:30 Memory and audit (40s).**
   - Approve a pending action and resolve the case: memory updates.
   - Show the timeline tab (every event), the SAR draft, and the answer file download.
   - Show the Portfolio tab with all 20 cases.
7. **4:10 Close (10s).** From an uncertain signal to a defensible action, with a complete record.
