# From an uncertain fraud signal to a defensible decision: an investigation agent on TigerGraph

*Technical write-up for the TigerGraph Agentic Fraud Investigation hackathon (HHGOA).*

> **Draft.** Everything here describes the design. No HHGOA results are reported yet. Development ran on clearly
> labelled synthetic data, and there is no answer key, so no accuracy is claimed. Fill in results after the benchmark
> run on the real dataset.

## What we built

Fraud analysts spend most of their time collecting evidence, and much less of it deciding. They pull transaction
history, check whether a device has been seen before, look for other accounts sharing it, read the policy, look up how
similar cases ended, write everything down, and then choose an action. We built an agent that does that loop end to end
on the IEEE-CIS-based HHGOA dataset.

The agent starts from a trigger: a model score, a customer report or an analyst request. It opens a case and
investigates it in a TigerGraph knowledge graph. It then decides whether it has enough evidence to act. If it doesn't,
it asks for the single piece of evidence most likely to settle the question. It records the next best action both
before and after that evidence arrives, and executes only what policy allows. Everything else goes to the right human
approver, and the agent explains every step.

## Architecture

- **TigerGraph (Savanna / Community Edition)** holds three layers:
  - transactions: `Transaction`, `Customer`, `Card`, `Device`, `Network`, `Email`, `Address`
  - the investigation layer: `FraudCase`, `CaseEvent`, `Pattern`, `PolicyChunk`, with `SIMILAR_TO`,
    `MATCHES_PATTERN` and `DESCRIBED_BY` edges
  - vector attributes for case and policy embeddings
- **GSQL installed queries** provide the investigation primitives:
  - `customer_profile`: the customer's behavioural baseline
  - `entity_links`: shared entities, ignoring hub entities
  - `ring_expand`: bounded multi-hop BFS
  - `related_cases`: graph-linked prior cases
  - `build_communities`: WCC over non-hub shared devices, networks and cards
  - `community_stats`

  Every query is point-in-time, so the agent never reads data from after the trigger.
- **TigerGraph MCP** is the only way the agent touches the graph at runtime. It runs installed queries, upserts
  `FraudCase` and `CaseEvent` vertices and their edges, and reads case and policy vectors. Bulk loading uses
  pyTigerGraph, because 590k rows don't belong in a tool protocol.
- **GraphRAG** splits the policy, pattern and regulatory documents into `PolicyChunk` vertices with embeddings. Each
  documented pattern becomes a `Pattern` vertex linked to its chunks. At decision time the agent retrieves context two
  ways: it expands from the pattern hypotheses through the graph, and it runs a vector search on the case's
  question. The LLM receives a compact, cited context pack rather than raw rows.
- **An LLM** (via tool use) plans the investigation, chooses tools, interprets the evidence, writes
  findings, and drafts the case summary, explanation and SAR narrative.
- **A deterministic decision engine** owns the numbers the LLM should not invent:
  - the calibrated P(fraud)
  - confidence
  - the stopping rule
  - the value of information for each evidence request
  - policy-routed next best actions
- **A Streamlit analyst console** shows the case progression. For each decision stage it shows the NBA with its
  approval route, plus a graph view of shared entities, pattern hypotheses, similar cases, the timeline, the SAR draft
  and the policy hits. Analysts approve or reject pending actions there and resolve cases.

## How TigerGraph is used

The graph is what turns a transaction score into an investigation. Take one transaction on a known card:
- **One hop:** does the customer's device or network also appear under other customers? `entity_links` answers this,
  and flags hub entities like `gmail.com` so they are ignored.
- **Several hops:** does the customer sit inside a cluster of accounts connected through shared devices and cards?
  `ring_expand` and the WCC communities answer this.
- **Memory:** did an analyst already confirm fraud on any of those connected entities? `related_cases` walks from the
  trigger to `FraudCase` vertices.

These answers become named signals, and the risk model learns their weights from the closed cases. We will report which
signals carry the most weight once the model has been fitted on the HHGOA data. The same graph then stores the agent's own case record, so the next investigation can
find it.

## Agentic capabilities

1. **It knows when to stop.** The evidence is enough when the posterior crosses an action threshold *and* confidence
   meets the policy minimum.
2. **It asks the right question, not every question.** Each outcome of each permitted evidence request has modelled
   probabilities under fraud and under legitimate activity. The agent picks the request with the highest probability
   of a decisive answer, net of customer friction, and it never contacts a customer who has already reported the
   charge. The NBA is recorded both *before* and *after* the evidence arrives.
3. **Policy-bounded autonomy.** Actions go through a permission matrix: auto, analyst, senior, compliance or never.
   Auto actions also have amount limits. If the LLM proposes something stronger than the engine recommends, it becomes
   a human-approval item, and mandated actions such as SAR filing cannot be dropped.
4. **Case memory.** Closed cases from months 1–4 become vectors with analyst outcomes. The risk model is fitted on
   them and similar cases are retrieved from them. When an analyst resolves a case in the UI, the model is refitted.
5. **Undocumented patterns.** If the posterior is high but no documented typology explains it, the case is labelled
   *unclassified (possible undocumented pattern)* rather than forced into the nearest bucket.
6. **Explainability.** Each case records its signal contributions in log-odds, the evidence with its likelihood
   ratios, the reason each request was made, a rationale and approval route for each action, the policy citations and
   the remaining uncertainty.

## What we learned

- The bank's risk score works as a *prior*, not a decision. How far graph context moves it on the HHGOA cases is still
  to be measured.
- Hub entities dominate naive graph features, so degree-aware filtering matters more than query cleverness.
- LLMs are excellent at interpretation and explanation, and unreliable as calculators. Splitting the work (the LLM
  plans and explains, the engine scores, policy gates) gave us both flexibility and auditability.
- Point-in-time correctness has to be built into every query from the start, or the evaluation lies.

## What we would improve with more time

- Replace the modelled likelihood ratios for customer and step-up responses with ratios learned from outcome data.
- Use TigerGraph GDS algorithms (Louvain, weighted PageRank from confirmed-fraud seeds) as additional signals, and
  temporal motifs for card testing.
- Add active learning: route the most informative uncertain cases to analysts first.
- Stream triggers from a queue and investigate in real time, before the money leaves.
