# Agentic Fraud Investigation on TigerGraph (HHGOA)

An AI agent that investigates card fraud from a trigger (risk signal, customer report or analyst request), opens and
progresses a case, gathers evidence from a TigerGraph knowledge graph via **TigerGraph MCP**, grounds itself in the bank's
policy/typologies with **GraphRAG**, learns from **prior case outcomes**, requests extra evidence only when it is worth it,
and recommends policy-routed **next best actions** with a full audit trail, SAR draft and explanation.

```mermaid
flowchart LR
  T[Trigger<br/>risk score / customer report / analyst] --> A
  subgraph Agent["Investigation agent (LLM + deterministic guardrails)"]
    A[LLM planner<br/>tool selection & narrative] -->|tool calls| TL
    TL[Tools] --> DE[Decision engine<br/>calibrated P(fraud), confidence,<br/>stop rule, value of information]
    DE --> PG[Policy gate<br/>permissions & approval routes]
  end
  TL -->|MCP: run_installed_query / add_nodes / add_edges| TG[(TigerGraph<br/>FraudGraph)]
  TL --> RAG[GraphRAG<br/>policy · patterns · regs]
  TL --> MEM[Case memory<br/>vectors + outcomes]
  TL --> EV[Controlled evidence<br/>customer validation · step-up · analyst info]
  PG --> ACT[Mock action APIs<br/>block / hold / warn / refund / SAR]
  PG --> CASE[Case record<br/>events · decisions · approvals]
  CASE -->|FraudCase, CaseEvent, Pattern edges| TG
  CASE --> MEM
  CASE --> UI[Streamlit analyst console]
```

## What happens in one investigation

1. **Trigger:** a case is opened with the trigger type and detail.
2. **Investigate (GSQL via MCP):** the agent pulls the following, all point-in-time (nothing after the trigger is read):
   - `txn_context`
   - `customer_profile` (the customer's behavioural baseline)
   - `entity_links` (other customers on the same device, network, card, address or email, ignoring hub entities)
   - `ring_expand` (bounded BFS across shared entities)
   - `community_stats` (WCC communities from `build_communities`)
   - `related_cases` (prior cases linked in the graph)
3. **Case memory:** finds the nearest prior investigations and their analyst outcomes. The match uses cosine similarity
   over standardized evidence vectors, with each signal weighted by its importance in the risk model.
4. **Assess:** a logistic model fitted on the closed cases (months 1–4) gives the calibrated P(fraud), with a
   contribution for each signal. It also gives:
   - pattern hypotheses: detectors mapped to the *documented* patterns, blended with precedent, with an undocumented
     pattern flagged when nothing fits
   - confidence (margin × coverage × agreement with precedent × pattern clarity × strength of the extra evidence)
   - the SAR test
5. **Uncertain?** If P(fraud) sits between the thresholds or confidence is low, the engine ranks the policy-permitted
   evidence requests by **expected value of information**: the chance each one produces a decisive answer, minus the
   friction it causes the customer.
   - The NBA is recorded *before* the request.
   - The response updates the posterior through likelihood ratios, and the NBA is recorded again *after* it.
   - If no request can resolve the case or the evidence budget is spent, the case is escalated.
6. **Act:** the policy gate executes only `auto` actions, through stubbed APIs. Everything else waits for the named
   approver (analyst, senior analyst or compliance). If the LLM proposes an action stronger than the engine's
   recommendation, that action is always routed to a human.
7. **Explain and record:** the agent writes the case summary, explanation, findings, SAR draft and full event log. It
   writes the `FraudCase`, `CaseEvent`, `Pattern`, `SIMILAR_TO` and `MATCHES_PATTERN` records to the graph and updates
   case memory. When an analyst resolves the case in the UI, the outcome is fed back and the model is refitted.

## Repository layout

| Path | What |
|---|---|
| `gsql/schema.gsql`, `gsql/queries.gsql` | Graph schema (transactions + entities + investigation layer) and installed queries (incl. WCC) |
| `fraud_agent/graph/mcp_client.py` | Long-lived TigerGraph MCP client used by the agent at runtime |
| `fraud_agent/graph/backend.py` | `TigerGraphBackend` (MCP) and `LocalBackend` (offline mirror for development/tests) |
| `fraud_agent/agent/orchestrator.py` | Agent loop (LLM tool use), tools, completeness check, execution, persistence |
| `fraud_agent/agent/assess.py` | Decision engine: posterior, confidence, stop rule, VOI, next best actions |
| `fraud_agent/agent/memory.py` | Case memory + risk model learned from closed cases |
| `fraud_agent/agent/patterns.py` | Pattern detectors, mapping to documented typologies, undocumented-pattern flag |
| `fraud_agent/agent/policy.py`, `config/policy_rules.yaml` | Permissions, approval routes, SAR rules, evidence outcome model |
| `fraud_agent/rag/` | GraphRAG over policy / pattern / regulatory documents |
| `fraud_agent/report.py` | Answer-file writer (JSON + Markdown per case) |
| `ui/app.py` | Analyst console |
| `config/datamap.yaml` | Dataset column/file mapping (edit here if names differ) |
| `scripts/validate_outputs.py` | **Structural** completeness check of answer files (not an accuracy measure) |
| `tests/mock_llm_loop.py` | Exercises the LLM tool loop and policy gate with a mocked client (no API key needed) |

## Current status (read this first)

| Item | State |
|---|---|
| HHGOA_IEEE dataset | **Not present in this workspace.** Only `data/SYNTH` exists, generated by `fraud_agent/data/synth.py`. The dataset schema, benchmark cases, policy text and answer format have **not** been read. |
| Answer format | **Provisional.** The dataset README defines the real format. Adapt `fraud_agent/report.py::answer()` once it is available. |
| Policy thresholds and approval routes | **Placeholder defaults** (`config/policy_rules.yaml`, `meta.status: placeholder_defaults`). Align them with the dataset's bank policy, then set `meta.status: aligned_with_dataset`. |
| TigerGraph | **No instance configured.** The GSQL has not yet been installed on a real server, so expect a round of syntax fixes on first install. The agent currently runs against `LocalBackend`, a labelled local mirror of the queries, and cases are **not** persisted to TigerGraph. |
| LLM | **No API key configured.** The tool loop is verified with a mocked client only, and runs use the deterministic planner. |
| Benchmark answers | `outputs/synthetic/answers/` holds 20 answers for the **synthetic** cases BM01–BM20. They are labelled synthetic in every file and are **not** HHGOA results. No accuracy is claimed: there is no answer key. |

## Stack choice

The workspace had no existing project for this challenge; sibling folders are unrelated Python/Streamlit apps.
We chose **Python**, because pyTigerGraph, the tigergraph-mcp server, the Anthropic SDK and the pandas/scikit-learn
tooling for IEEE-CIS are all Python. We chose **Streamlit** so the analyst UI needs no separate frontend build.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env                              # fill in values; .env is git-ignored
```

### Environment variables

| Variable | Purpose |
|---|---|
| `DATA_DIR` | Folder holding the HHGOA_IEEE files (default `./data/HHGOA_IEEE`) |
| `GRAPH_BACKEND` | `tigergraph` (MCP, the real path) or `local` (offline mirror, clearly labelled) |
| `TG_HOST`, `TG_GRAPHNAME` | e.g. `https://<id>.i.tgcloud.io`, `FraudGraph` |
| `TG_SECRET` *or* `TG_USERNAME`/`TG_PASSWORD` | Savanna: create a secret in the admin portal |
| `TG_RESTPP_PORT`, `TG_GS_PORT` | `443` for Savanna; `9000` / `14240` for Community Edition |
| `ANTHROPIC_API_KEY` | Enables the LLM agent loop (optional; without it the deterministic planner runs) |
| `LLM_MODEL`, `LLM_EFFORT` | Model id from your LLM provider (required to enable the LLM loop); effort default `medium` |
| `EVIDENCE_MODE` | `dataset` (dataset responses, falling back to seeded simulation), `simulated` or `interactive` |
| `OUTPUT_DIR` | Default `outputs/hhgoa` for real data, `outputs/synthetic` for synthetic data |
| `MAX_HUB`, `MAX_RING` | Hub-entity degree cutoff (60) and ring expansion cap (500) |

### Steps

1. Put the HHGOA_IEEE folder at `data/HHGOA_IEEE` and run `python scripts/inspect_dataset.py`.
   - It prints the README, the files, and how `config/datamap.yaml` resolves each column. Fix any mapping there.
   - Update `report.py::answer()` to the README's answer format.
   - Align `config/policy_rules.yaml` with the bank policy.
2. Set up TigerGraph: Savanna with auto-stop and auto-start enabled, or Community Edition. Then run
   `python scripts/setup_graph.py --all`, which does:
   - `--schema`: `gsql/schema.gsql`, plus native vector attributes when the server supports them
   - `--load`: bulk upsert through pyTigerGraph (transactions, customers, entities with degree, edges)
   - `--queries`: install `gsql/queries.gsql`
   - `--communities`: run `build_communities` (WCC)
   - `--memory`: compute evidence vectors for the closed cases, fit the risk model, upsert `FraudCase` vertices and vectors
   - `--docs`: `PolicyChunk` and `Pattern` vertices with `DESCRIBED_BY` edges (GraphRAG)
3. Run `python scripts/run_benchmark.py` (add `--no-llm` for the deterministic planner). Then run
   `python scripts/validate_outputs.py`, which checks structure only.
4. Run `streamlit run ui/app.py`.

### TigerGraph MCP

At runtime the agent starts the `tigergraph-mcp` server (installed from requirements) over stdio, with the `TG_*`
environment variables. It calls these tools:
- `tigergraph__run_installed_query`
- `tigergraph__add_nodes`
- `tigergraph__add_edges`
- `tigergraph__get_nodes`

See `fraud_agent/graph/mcp_client.py`. Every call is recorded. To expose the same graph to another MCP client, such as
an MCP desktop client:

```json
{"mcpServers": {"tigergraph": {"command": "tigergraph-mcp",
  "env": {"TG_HOST": "https://<id>.i.tgcloud.io", "TG_GRAPHNAME": "FraudGraph", "TG_SECRET": "<secret>",
          "TG_RESTPP_PORT": "443", "TG_GS_PORT": "443"}}}}
```

### Offline / synthetic mode

```bash
python -m fraud_agent.data.synth data/SYNTH
```

Then set `DATA_DIR=./data/SYNTH` and `GRAPH_BACKEND=local`. Every output is marked **SYNTHETIC**, goes to
`outputs/synthetic/`, and the UI shows a red banner. Synthetic results say nothing about performance on HHGOA.

## Provenance and honesty rules built in

- **Findings are typed:**
  - `observed`: read directly from the data
  - `computed`: a derived signal or score
  - `hypothesis`: a model interpretation

  Each finding carries source identifiers: transaction, customer, entity, case and policy chunk IDs.
- **Evidence provenance.** Every evidence section records the query that produced it, whether it ran through
  TigerGraph MCP or the local mirror, and the IDs involved.
- **Answer metadata.** Each answer file carries a `meta` block recording:
  - the dataset label
  - whether the case was persisted to the graph
  - the policy-config status
  - the agent mode (LLM or deterministic)
  - the source of each piece of additional evidence
  - that all actions are simulated
- **All actions are simulated.** `fraud_agent/agent/actions.py` only appends to `action_log.jsonl`; it never contacts
  an external system.

## Controls

- **Permissions.** Only `auto` actions are executed, and only as simulations. `block_transaction` above $1,000 goes to
  an analyst. Card blocks need an analyst, account blocks need a senior analyst, and SAR filing needs compliance.
  These thresholds are the placeholder defaults described above.
- **Evidence requests are policy-gated.** A request is refused when:
  - the assessment is already decisive
  - the request doesn't apply to the trigger (a customer who reported a charge isn't asked to "validate" it)
  - it was already used
  - the evidence-round budget is spent
- **The LLM is not the classifier or the policy authority.** It chooses tools and writes explanations. Risk comes from
  the calibrated engine and permissions from the deterministic gate. Anything the LLM adds beyond the engine's
  recommendation goes to a human. It cannot drop `file_sar`, `escalate_to_analyst` or `create_case`.
- **Audit trail.** Every MCP call, tool call, assessment, status change, approval and graph write is in the case event
  log.
