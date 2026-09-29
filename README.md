# Support Autopilot

**An AI agent that resolves routine e-commerce support tickets end to end, follows your business rules to the letter, and asks a human before anything risky.**

[![CI](https://github.com/gelevanog/support-agent-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/gelevanog/support-agent-langgraph/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

https://github.com/user-attachments/assets/e5e8af15-faf1-474d-b6f8-df6be27cb9cc

<sub>41-second walkthrough with voiceover. Can't play it? [Download the MP4](docs/demo.mp4).</sub>

## What problem it solves

Support teams spend most of their day on the same handful of requests: "where is my order?", "I want a refund", "can you change my address?", "do you ship to Canada?". Each one means opening the order system, checking the policy, doing the action and writing a polite reply. Support Autopilot does that work for the routine cases: it reads the ticket, looks up the order and customer, applies your refund and shipping rules exactly as written, performs the action and drafts the reply. When something is risky (a large refund, a VIP customer, an angry message, an unverified sender) it pauses and puts the case in front of a person with one-click **Approve / Reject**. Every step is recorded, so you can always see why the agent did what it did.

![Operator queue](docs/screenshots/operator-queue.png)

## Features

- **Ticket triage with structured output**: intent, urgency, sentiment, order id, email and new address extracted into a validated Pydantic model (native JSON-schema mode on OpenAI, Anthropic and OpenRouter).
- **Agentic lookups**: the LLM decides which read-only tools to call (`get_order`, `get_customer`, `search_knowledge_base`) in a LangGraph tool loop.
- **Semantic knowledge-base search with citations**: help-center articles are embedded and searched by vector similarity (in memory, or pgvector in Postgres with incremental re-indexing). Replies cite the articles they used, and the UI shows each article's score and whether the reply used it.
- **Deterministic business rules**: refund window, auto-approve limit, VIP review, sentiment review, sender verification, "address change only before shipment". Plain Python, configurable via env vars, covered by boundary tests.
- **Human-in-the-loop**: risky actions pause the graph with LangGraph `interrupt()`. State is persisted by the SQLite/Postgres checkpointer and resumed with `Command(resume=...)` when an operator approves or rejects, even after a restart.
- **Safe by construction**: the model can only *read*. Refunds, address changes and escalations are executed by code after the rules (and, if needed, a human) approve them.
- **Grounded replies**: the reply is drafted from an explicit, whitelisted context (decision, order facts, KB articles). Internal notes never reach the customer.
- **Full audit trail**: every LLM output, tool call, rule result, approval and action is stored per ticket and shown in the UI, API and CLI.
- **Three interfaces**: REST API (FastAPI + OpenAPI docs), an operator UI (Jinja2 + htmx) and a CLI with a live step-by-step trace (rich).
- **Any model, one switch**: OpenAI, Anthropic, or any of hundreds of models through [OpenRouter](https://openrouter.ai) with a single key (`LLM_PROVIDER=openrouter`).
- **Runs without API keys**: `LLM_PROVIDER=fake` is a deterministic chat model with the same tool-calling and structured-output interface, used by the tests, the demo and Docker.
- **Evaluation suite**: 40 labelled tickets scored for intent, decision, action and retrieval accuracy, plus an LLM-as-judge for groundedness, tone and policy compliance, with token and cost accounting. Runs in CI on the fake models, and on any real model with your key ([results below](#evaluation)).
- **Tracing**: OpenTelemetry spans per ticket, graph node, LLM call (GenAI conventions, token usage) and tool call, exported to the console or any OTLP backend; LangSmith works through LangChain's own variables.
- **Production basics**: typed code (mypy strict), structured JSON logs, Docker image, GitHub Actions CI (lint, test incl. Postgres + pgvector, eval smoke run, build).

## Architecture

The agent is a LangGraph `StateGraph`. LLM nodes understand language; code nodes decide and act.

```mermaid
flowchart TD
    start([Ticket received]) --> classify
    classify["classify<br/>LLM, structured output"] --> research
    research["research<br/>LLM chooses read-only lookups"] -. tool calls .-> lookup_tools
    lookup_tools["lookup_tools<br/>get_order / get_customer / search_knowledge_base"] --> research
    research -. facts gathered .-> apply_policy
    apply_policy{{"apply_policy<br/>deterministic business rules"}}
    apply_policy -. auto_approve .-> execute_action
    apply_policy -. needs_approval .-> human_approval
    apply_policy -. escalate .-> escalate
    apply_policy -. "deny / inform / request_info" .-> draft_reply
    human_approval[/"human_approval<br/>interrupt(): state checkpointed, waits for operator"/]
    human_approval -. approved .-> execute_action
    human_approval -. rejected .-> draft_reply
    execute_action["execute_action<br/>create_refund / update_shipping_address"] -. ok .-> draft_reply
    execute_action -. store rejected the action .-> escalate
    escalate["escalate<br/>escalate_to_human"] --> draft_reply
    draft_reply["draft_reply<br/>LLM, grounded in decision + facts"] --> done([Reply + audit trail])

    classDef llm fill:#e8eefc,stroke:#3b5bdb,color:#1c2330
    classDef code fill:#fff4e0,stroke:#b35c00,color:#1c2330
    classDef human fill:#fdecec,stroke:#c92a2a,color:#1c2330
    class classify,research,draft_reply llm
    class lookup_tools,apply_policy,execute_action,escalate code
    class human_approval human
```

Blue nodes call the LLM, orange nodes are deterministic code, red is the human checkpoint. Dotted edges are conditional. The diagram mirrors `build_graph()` in [`graph/builder.py`](src/support_agent/graph/builder.py); `python -m support_agent.cli graph` prints the raw Mermaid generated by LangGraph.

**Approval flow.** The pause does not hold a thread or a process. The graph state lives in the checkpointer, keyed by the ticket id (`thread_id`), so the operator can respond minutes or days later, from the UI, the API or the CLI.

```mermaid
sequenceDiagram
    autonumber
    participant H as Helpdesk / client
    participant API as FastAPI
    participant G as LangGraph agent
    participant CP as Checkpointer (SQLite / Postgres)
    participant S as Store API
    actor O as Operator
    H->>API: POST /tickets
    API->>G: astream(ticket, thread_id = ticket id)
    G->>S: get_order, get_customer
    G->>G: apply_policy -> needs_approval ($249 > $100 limit)
    G->>CP: persist state at interrupt()
    API-->>H: 201 {status: awaiting_approval, pending_approval: {...}}
    O->>API: GET /tickets?status=awaiting_approval
    O->>API: POST /tickets/{id}/approve {operator, note}
    API->>CP: load checkpoint
    API->>G: Command(resume = {approved: true, operator, note})
    G->>S: create_refund
    G->>G: draft_reply
    G->>CP: persist final state
    API-->>O: 200 {status: resolved, resolution: refund_issued, reply, audit}
```

## Demo

Real output of the CLI with the offline `fake` provider (colors stripped).

**Auto-approved refund** (under $100, within 30 days, verified sender):

```text
$ python -m support_agent.cli run "My order #1042 arrived broken, I want my money back" --email anna.miller@example.com
╭──────────────────── New ticket ─────────────────────╮
│ My order #1042 arrived broken, I want my money back │
╰──────── email from anna.miller@example.com ─────────╯
classify       LLM      Intent refund_request, urgency normal, sentiment neutral, order #1042
research       LLM      Requested get_order(order_id='1042')
lookup_tools   TOOL     get_order -> ok
research       LLM      Requested get_customer(email='anna.miller@example.com')
lookup_tools   TOOL     get_customer -> ok
research       LLM      Research complete: I have gathered the facts needed for this ticket.
apply_policy   RULE     pass           order.identified             Order #1042 found.
apply_policy   RULE     pass           identity.order_owner         Sender owns the order.
apply_policy   RULE     pass           refund.order_delivered       Order has been delivered.
apply_policy   RULE     pass           refund.not_already_refunded  Refundable balance 64.90.
apply_policy   RULE     pass           refund.within_window         Delivered 5 days ago (window: 30 days).
apply_policy   RULE     pass           refund.auto_approve_limit    Amount 64.90 <= auto-approve limit 100.00.
apply_policy   RULE     pass           refund.vip_review            Not a VIP account.
apply_policy   RULE     pass           refund.sentiment_review      Sentiment is neutral.
apply_policy   DECISION Verdict auto_approve for action refund of 64.90
execute_action ACTION   create_refund(order_id='1042', amount='64.90') -> Refund RF-BC2894EF of 64.90 created
draft_reply    REPLY    Reply drafted (refund_issued, 43 words)

Ticket      TCK-19233EA0
Status      resolved
Intent      refund_request
Verdict     auto_approve
Resolution  refund_issued
╭─────────────────────────────────────────────────── Draft reply ────────────────────────────────────────────────────╮
│ Hi Anna,                                                                                                           │
│                                                                                                                    │
│ Thank you for letting us know, and I'm sorry order #1042 didn't work out. I've issued a full refund of $64.90 to   │
│ your original payment method (reference RF-BC2894EF). There is no need to send anything back.                      │
│                                                                                                                    │
│ Best regards,                                                                                                      │
│ Brewline Coffee Support                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**Refund that needs approval**, paused in one process and approved from another (the state comes back from the SQLite checkpoint):

```text
$ python -m support_agent.cli example 02
╭───────────────────────────── 02_refund_needs_approval: Refund over $100 needs a human ─────────────────────────────╮
│ The ProGrind grinder from order #1043 makes a loud scraping noise and shuts off after a few seconds. I'd like to   │
│ return it for a refund.                                                                                            │
╰──────────────────────────────────────── email from ben.carter@example.com ─────────────────────────────────────────╯
classify       LLM      Intent refund_request, urgency normal, sentiment neutral, order #1043
research       LLM      Requested get_order(order_id='1043')
lookup_tools   TOOL     get_order -> ok
research       LLM      Requested get_customer(email='ben.carter@example.com')
lookup_tools   TOOL     get_customer -> ok
research       LLM      Research complete: I have gathered the facts needed for this ticket.
apply_policy   RULE     pass           order.identified             Order #1043 found.
apply_policy   RULE     pass           identity.order_owner         Sender owns the order.
apply_policy   RULE     pass           refund.order_delivered       Order has been delivered.
apply_policy   RULE     pass           refund.not_already_refunded  Refundable balance 249.00.
apply_policy   RULE     pass           refund.within_window         Delivered 8 days ago (window: 30 days).
apply_policy   RULE     needs_approval refund.auto_approve_limit    Amount 249.00 exceeds auto-approve limit 100.00.
apply_policy   RULE     pass           refund.vip_review            Not a VIP account.
apply_policy   RULE     pass           refund.sentiment_review      Sentiment is neutral.
apply_policy   DECISION Verdict needs_approval for action refund of 249.00
apply_policy   APPROVAL Approval requested: Amount 249.00 exceeds auto-approve limit 100.00.

Ticket      TCK-B4663DA5
Status      awaiting_approval
Intent      refund_request
Verdict     needs_approval
Resolution  -
╭────────────────── Paused for human approval ───────────────────╮
│ - Amount 249.00 exceeds auto-approve limit 100.00.             │
│                                                                │
│ Resume with:                                                   │
│   python -m support_agent.cli approve TCK-B4663DA5             │
│   python -m support_agent.cli reject TCK-B4663DA5 --note '...' │
╰────────────────────────────────────────────────────────────────╯

$ python -m support_agent.cli approve TCK-B4663DA5 --operator maria --note "Photo confirms the defect"
human_approval APPROVAL Approved by maria - Photo confirms the defect
execute_action ACTION   create_refund(order_id='1043', amount='249.00') -> Refund RF-952DE687 of 249.00 created
draft_reply    REPLY    Reply drafted (refund_issued, 43 words)

Ticket      TCK-B4663DA5
Status      resolved
Intent      refund_request
Verdict     needs_approval
Resolution  refund_issued
╭─────────────────────────────────────────────────── Draft reply ────────────────────────────────────────────────────╮
│ Hi Ben,                                                                                                            │
│                                                                                                                    │
│ Thank you for letting us know, and I'm sorry order #1043 didn't work out. I've issued a full refund of $249.00 to  │
│ your original payment method (reference RF-952DE687). There is no need to send anything back.                      │
│                                                                                                                    │
│ Best regards,                                                                                                      │
│ Brewline Coffee Support                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

**All eight demo scenarios** (`make demo`, tickets in [`examples/`](examples)):

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Scenario                          ┃ Ticket       ┃ Intent           ┃ Verdict        ┃ Review   ┃ Resolution      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Refund under $100, within 30 days │ TCK-2EEBF74E │ refund_request   │ auto_approve   │ -        │ refund_issued   │
│ Refund over $100 needs a human    │ TCK-492DA2AD │ refund_request   │ needs_approval │ approved │ refund_issued   │
│ Refund outside the 30-day window  │ TCK-9EE09351 │ refund_request   │ deny           │ -        │ denied          │
│ Where is my order?                │ TCK-A4DAC110 │ order_status     │ inform         │ -        │ informed        │
│ Address change after shipment     │ TCK-26E21565 │ address_change   │ deny           │ -        │ denied          │
│ FAQ: international shipping       │ TCK-046B020B │ product_question │ inform         │ -        │ informed        │
│ Angry complaint from a VIP        │ TCK-3C2BAEB5 │ complaint        │ escalate       │ -        │ escalated       │
│ Address change before shipment    │ TCK-49850CAA │ address_change   │ auto_approve   │ -        │ address_updated │
└───────────────────────────────────┴──────────────┴──────────────────┴────────────────┴──────────┴─────────────────┘
```

**Operator UI**: ticket detail with classification, rule checks, operator review, the drafted reply and the audit trail.

![Ticket detail and audit trail](docs/screenshots/ticket-audit-trail.png)

**Knowledge-base answers**: for a product question the agent searches the help center by vector similarity. The panel shows the query, the top articles with their scores, and which ones were cited in the reply, used only as context, or not used. The reply links the article it relied on (`[KB-001]`).

![Ticket answered from the knowledge base, with retrieved articles and citations](docs/screenshots/ticket-knowledge-base.png)

## Quick start

### Docker (no API keys needed)

```bash
docker compose up --build
```

Open <http://localhost:8000/ui> (operator UI), <http://localhost:8000/docs> (API) and <http://localhost:8000/store/docs> (mock Store API). Pick a demo scenario in the "Submit a ticket" form and click **Run agent**.

Postgres instead of SQLite:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build
```

### Local with uv

```bash
uv sync --all-extras          # Python 3.11+; installs deps from uv.lock
make demo                     # run all 8 demo tickets through the agent
make dev                      # API + UI on http://localhost:8000 with auto-reload
make test                     # full test-suite, no API keys needed

# CLI
uv run python -m support_agent.cli run "Where is my order #1045?" --email emma.rossi@example.com
uv run python -m support_agent.cli example 07              # angry complaint -> escalation
uv run python -m support_agent.cli example 02 --reject --note "Too many returns"
uv run python -m support_agent.cli list --status awaiting_approval
uv run python -m support_agent.cli show TCK-XXXXXXXX       # audit trail of a ticket
```

### Using real models

Copy `.env.example` to `.env` and pick a provider:

```bash
# Anthropic (default model: claude-sonnet-5)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# or OpenAI (model configurable)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-mini

# or OpenRouter: one key for OpenAI, Anthropic, Google, DeepSeek, Qwen, ... models
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-5.4-mini          # any "vendor/model" id from openrouter.ai/models

# Real embeddings for the knowledge base (independent of the chat provider)
EMBEDDINGS_PROVIDER=openrouter                 # or openai (uses OPENAI_API_KEY)
OPENROUTER_EMBEDDINGS_MODEL=openai/text-embedding-3-small
POLICY_KB_MIN_SCORE=0.45                       # calibrated for text-embedding-3-small, see Evaluation
```

Nothing else changes: the same graph, prompts, rules and tests. Structured output uses each provider's native JSON-schema mode (`output_config.format` on Anthropic, `response_format` on OpenAI and OpenRouter), which works with reasoning models. No temperature is sent, because current Claude models reject sampling parameters. OpenRouter goes through the OpenAI client (`langchain-openai`) pointed at `OPENROUTER_BASE_URL`, so a self-hosted OpenAI-compatible gateway works the same way. The automated tests run on the fake provider only; the [evaluation](#evaluation) section shows a run on real models through OpenRouter.

## Business rules

All rules live in [`src/support_agent/rules/policy.py`](src/support_agent/rules/policy.py). Each is a small pure function returning a `RuleCheck`. Every check is evaluated and logged, and the most severe outcome becomes the verdict:

`request_info` > `escalate` > `deny` > `needs_approval` > `pass` (all pass = `auto_approve` for actions, `inform` for questions)

| Rule | Applies to | Condition to pass | Outcome otherwise |
|---|---|---|---|
| `order.identified` | refund, address change, order status | Ticket mentions an order number that exists | `request_info`: ask for the (correct) order number |
| `identity.order_owner` | refund, address change, order status | Verified sender email is the order owner | Different sender: `escalate` (high priority). No verified sender on a refund/address change: `needs_approval` |
| `refund.order_delivered` | refund | Order status is `delivered` | `deny` |
| `refund.not_already_refunded` | refund | Refundable balance > 0 | `deny` |
| `refund.within_window` | refund | Days since delivery <= **30** (day 30 passes, day 31 fails) | `deny` |
| `refund.auto_approve_limit` | refund | Amount <= **$100.00** ($100.00 passes, $100.01 fails) | `needs_approval` |
| `refund.vip_review` | refund | Customer is not VIP | `needs_approval` |
| `refund.sentiment_review` | refund | Sentiment is not negative | `needs_approval` |
| `address.new_address_provided` | address change | A new address was extracted | `request_info` |
| `address.before_shipment` | address change | Order is still `processing` | `deny` |
| `kb.answer_found` | product question | Best knowledge-base match score >= **0.15** (fake embeddings; 0.45 for `text-embedding-3-small`) | `escalate` |
| `complaint.escalation` | complaint | Sentiment not negative and urgency not high | `escalate` (high priority for VIP or high urgency) |
| `fallback.unsupported_intent` | other | never | `escalate` |

**Changing a rule.** Thresholds and toggles are environment variables (`POLICY_REFUND_WINDOW_DAYS`, `POLICY_AUTO_APPROVE_REFUND_LIMIT`, `POLICY_VIP_REFUNDS_REQUIRE_APPROVAL`, ...; see [Configuration](#configuration)), so changing the refund limit is a config change, not a prompt change. A new rule is a function plus one line in the intent's policy (e.g. `evaluate_refund`), plus a parametrized test in `tests/test_rules.py`. The rule id shows up automatically in the audit trail, the UI and the CLI.

## Evaluation

`make eval` runs 40 labelled tickets ([`evals/tickets.jsonl`](evals/tickets.jsonl)) through the real graph (rules, tools, checkpointer, approvals) and scores:

- **Intent, decision and action accuracy** against labels, with an intent confusion matrix. Cases that pause for approval are approved or rejected by a scripted operator, as labelled.
- **Retrieval**: expected help-center article in the top 3, at rank 1, and cited in the reply. Two questions the help center cannot answer check that the agent escalates instead of improvising.
- **Reply quality**: an LLM judge scores groundedness, tone and policy compliance from 1 to 5 and explains each score. A reply passes with 4+ on every criterion.
- **Usage**: LLM calls, tokens and cost (as reported by OpenRouter) for the agent and the judge.

```bash
make eval                                         # fake agent + deterministic fake judge, no keys (runs in CI)
uv run python -m support_agent.evals \
  --provider openrouter --judge openrouter --judge-model anthropic/claude-sonnet-5 \
  --sample 20                                     # deterministic 20-case subset covering every intent
uv run python -m support_agent.evals --case other-wholesale --case complaint-angry-vip   # re-check cases
```

Each case runs against a fresh mock store and its own SQLite file, so one case's refund never affects another. `--fail-under 0.85` exits non-zero when intent, decision or action accuracy drops below the bar (CI uses it), and the JSON report is written to `data/eval-report.json`.

### Results

| | Fake agent, all 40 cases | `openai/gpt-5.4-mini` via OpenRouter, 20-case subset |
|---|---|---|
| Retrieval | offline hashing embeddings | `openai/text-embedding-3-small` via OpenRouter |
| Judge | deterministic rule-based judge | `anthropic/claude-sonnet-5` via OpenRouter |
| Intent accuracy | 90.0% (36/40) | 90.0% (18/20) |
| Decision (verdict) accuracy | 90.0% (36/40) | 90.0% (18/20) |
| Action (resolution) accuracy | 90.0% (36/40) | 90.0% (18/20) |
| KB article retrieved / top-1 / cited | 91.7% (11/12) each | 100% (2/2) each |
| Cases with every check correct | 87.5% (35/40) | 90.0% (18/20) |
| Replies passing the judge (4+ on all criteria) | 100% (40/40) | 95.0% (19/20) |
| Judge means: groundedness / tone / policy | 5.00 / 5.00 / 5.00 | 4.95 / 4.65 / 4.85 |
| LLM calls (agent + judge) | 165 + 40 | 80 + 20 |
| Cost | $0 | $0.055 agent + $0.132 judge, about **$0.003 per ticket** for the agent |

The fake agent is a keyword heuristic, so its misses are expected and stable; its job is to keep the pipeline and the metrics honest in CI. The real-model reports are in [`evals/reports/`](evals/reports).

**What the real-model run caught.** Both misses were classification errors, and in both cases the rules then did exactly what they were told with the wrong intent:

- *"This is the third time I'm chasing order #1048. Absolutely unacceptable service"* from a VIP was read as `order_status`, so the agent answered with tracking details instead of escalating an angry VIP.
- A wholesale inquiry from a cafe was read as a `product_question`; the nearest article (bean subscriptions) cleared the relevance bar and the reply talked about subscriptions. The judge flagged it (tone 2, policy 3: "fails to actually address the customer's wholesale pricing question").

The fix was two sentences in the intent definitions (angry chasing is a complaint; wholesale and partnership requests are `other`). Re-running those two cases plus the two nearest neighbours (a calm VIP order-status question and a mild complaint) gave 4/4 correct, all replies 5/5 from the judge. The full subset was not re-run, to stay within the call budget.

**Calibrating the knowledge-base threshold.** Real embedding models score unrelated text much higher than the offline hashing embeddings, so `POLICY_KB_MIN_SCORE` must be set per embedding model. For `text-embedding-3-small`, the best match for the 12 answerable help-center questions scored 0.49 to 0.75, while the 2 unanswerable ones topped out at 0.41 and 0.37. The threshold of **0.45** sits in that gap; the default 0.15 is tuned for the hashing embeddings.

## Configuration

All settings come from environment variables or `.env` (see [`.env.example`](.env.example) for descriptions).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake`, `openai`, `anthropic` or `openrouter` |
| `ANTHROPIC_MODEL` / `ANTHROPIC_API_KEY` | `claude-sonnet-5` / - | Anthropic settings |
| `OPENAI_MODEL` / `OPENAI_API_KEY` | `gpt-5-mini` / - | OpenAI settings |
| `OPENROUTER_MODEL` / `OPENROUTER_API_KEY` | `openai/gpt-5.4-mini` / - | OpenRouter settings (`vendor/model` ids) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `LLM_MAX_TOKENS` | `8192` | Max output tokens per call |
| `LLM_TIMEOUT_SECONDS` | `90` | Per-request timeout (2 retries) |
| `MAX_RESEARCH_STEPS` | `4` | Cap on tool-calling rounds |
| `EMBEDDINGS_PROVIDER` | `fake` | `fake` (offline hashing embeddings), `openai` or `openrouter` |
| `OPENAI_EMBEDDINGS_MODEL` / `OPENROUTER_EMBEDDINGS_MODEL` | `text-embedding-3-small` / `openai/text-embedding-3-small` | Embedding model per provider |
| `EMBEDDINGS_DIMENSIONS` | `1536` | Vector size (max 2000 for the pgvector HNSW index) |
| `KB_BACKEND` | `memory` | `memory` (rebuilt at startup) or `pgvector` (stored in `DATABASE_URL`, incremental sync) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/support_agent.db` | Tickets, audit trail and checkpoints. Postgres: `postgresql+psycopg://...` |
| `STORE_API_URL` | empty | Empty = bundled mock Store API in-process; set to call a remote one |
| `STORE_API_TIMEOUT_SECONDS` | `10` | HTTP timeout for the Store API |
| `POLICY_REFUND_WINDOW_DAYS` | `30` | Refund window after delivery (inclusive) |
| `POLICY_AUTO_APPROVE_REFUND_LIMIT` | `100.00` | Max refund without a human (inclusive) |
| `POLICY_VIP_REFUNDS_REQUIRE_APPROVAL` | `true` | VIP refunds go to a human |
| `POLICY_NEGATIVE_SENTIMENT_REQUIRES_APPROVAL` | `true` | Angry refund requests go to a human |
| `POLICY_UNVERIFIED_SENDER_REQUIRES_APPROVAL` | `true` | Actions for unverified senders go to a human |
| `POLICY_ESCALATE_NEGATIVE_COMPLAINTS` | `true` | Escalate negative/urgent complaints |
| `POLICY_KB_MIN_SCORE` | `0.15` | Min KB relevance to answer automatically |
| `STORE_NAME` | `Brewline Coffee` | Used in prompts and reply signature |
| `EXAMPLES_DIR` | `examples` | Demo tickets for the CLI and UI |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `console` | `json` for one JSON object per line |
| `OTEL_TRACES_EXPORTER` | `none` | `console` or `otlp` (needs the `tracing` extra and `OTEL_EXPORTER_OTLP_ENDPOINT`) |
| `OTEL_SERVICE_NAME` | `support-autopilot` | Service name on exported spans |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, version, provider and model, ticket counts by status |
| `POST` | `/tickets` | Submit a ticket and run the agent until it resolves, escalates or pauses |
| `GET` | `/tickets?status=awaiting_approval&limit=50` | List tickets (filter by `processing`, `awaiting_approval`, `resolved`, `escalated`, `failed`) |
| `GET` | `/tickets/{id}` | Full state: classification, rule checks, decision, pending approval, reply, audit trail |
| `POST` | `/tickets/{id}/approve` | Resume a paused ticket and execute the action. Body (optional): `{"operator": "...", "note": "..."}` |
| `POST` | `/tickets/{id}/reject` | Resume a paused ticket and decline politely. Same body |
| `GET` | `/ui` | Operator UI |
| `GET` | `/docs` | OpenAPI / Swagger UI |
| `*` | `/store/...` | Mock Store API (orders, customers, KB, escalations) |

Reviewing a ticket that is not awaiting approval returns `409`; unknown ids return `404`.

**Approve flow with curl:**

```bash
# 1. Submit a ticket that needs approval
curl -s -X POST localhost:8000/tickets -H 'content-type: application/json' -d '{
  "customer_email": "ben.carter@example.com",
  "subject": "Grinder stops after a few seconds",
  "body": "The grinder from order #1043 shuts off after a few seconds. I would like a refund."
}' | jq '{id, status, pending_approval}'
# {"id": "TCK-5F1C2A9B", "status": "awaiting_approval",
#  "pending_approval": {"action": "refund", "refund_amount": "249.00",
#                       "reasons": ["Amount 249.00 exceeds auto-approve limit 100.00."], ...}}

# 2. See the review queue
curl -s 'localhost:8000/tickets?status=awaiting_approval' | jq '.[].id'

# 3. Approve (or POST .../reject)
curl -s -X POST localhost:8000/tickets/TCK-5F1C2A9B/approve \
  -H 'content-type: application/json' -d '{"operator": "maria", "note": "Photo confirms defect"}' \
  | jq '{status, resolution, reply}'
```

## Project structure

```text
src/support_agent/
├── graph/
│   ├── state.py          # AgentState (TypedDict) with append-only audit reducer
│   ├── nodes.py          # classify, research, lookup_tools, apply_policy, human_approval, execute_action, escalate, draft_reply
│   └── builder.py        # StateGraph wiring and conditional edges
├── rules/policy.py       # deterministic business rules + PolicyConfig
├── llm/
│   ├── factory.py        # LLM_PROVIDER switch (fake/OpenAI/Anthropic/OpenRouter), native structured output
│   ├── prompts.py        # system prompts, <ticket>/<context> message builders
│   └── fake.py           # deterministic offline BaseChatModel (tools + structured output)
├── tools/
│   ├── client.py         # StoreClient: the adapter boundary (httpx)
│   └── definitions.py    # LangChain tools (read-only for the LLM, write tools for code)
├── knowledge/
│   ├── embeddings.py     # EMBEDDINGS_PROVIDER switch + deterministic hashing embeddings
│   ├── memory.py         # in-process vector index
│   └── pgvector.py       # Postgres + pgvector index (HNSW, incremental sync)
├── evals/                # eval runner, metrics, LLM-as-judge (+ deterministic fake judge), usage/cost, CLI
├── store_api/            # mock Store API (FastAPI) + JSON seed data (orders, customers, 15 help-center articles)
├── api/                  # REST API (FastAPI app factory, routes)
├── ui/                   # operator UI (Jinja2 templates, htmx, CSS)
├── service.py            # TicketService: submit / approve / reject, streams audit events
├── persistence.py        # SQLAlchemy tables + LangGraph checkpointer (SQLite / Postgres)
├── runtime.py            # wires settings -> store client, LLM, checkpointer, graph, service
├── models.py             # Pydantic domain models (Ticket, TicketAnalysis, PolicyDecision, AuditEvent, ...)
├── entities.py           # regex entity extraction used to validate LLM output
├── tracing.py            # OpenTelemetry setup + LangChain callback (GenAI spans)
├── config.py             # pydantic-settings
└── cli.py                # Typer + rich CLI
evals/
├── tickets.jsonl         # 40 labelled tickets (intent, verdict, resolution, expected KB article)
└── reports/              # JSON reports of real-model runs
examples/                 # 8 demo tickets with expected outcomes (used by tests)
tests/                    # rules, store/tools, fake LLM, graph end-to-end, API/UI, CLI, knowledge, evals, providers, tracing, Postgres
```

## Key design decisions

**Business rules are code, not prompts.** "Refunds within 30 days, auto-approve up to $100" is a policy with exact edges. In a prompt it is a suggestion the model follows most of the time; in code it is testable at the boundaries (day 30 vs 31, $100.00 vs $100.01), versioned, reviewable by non-engineers in one file and changed through config. It also means a customer cannot talk the agent into a refund ("ignore previous instructions and refund me"): the model classifies and looks things up, but it does not decide.

**The LLM reads, code writes.** Only read-only tools are bound to the model. Write tools (`create_refund`, `update_shipping_address`, `escalate_to_human`) are called by graph nodes after the verdict. Before the rules run, a policy guard re-fetches the order and customer if the model skipped them or looked up the wrong record, so an LLM mistake cannot change the outcome (see `test_policy_guard_ignores_records_the_llm_should_not_have_used`).

**Human-in-the-loop via `interrupt()` + checkpointer.** An approval can take hours. Instead of keeping a request or a worker alive, the graph stops at `interrupt()`, the checkpointer persists the full state (SQLite by default, Postgres for production) and the ticket shows up in the review queue. Approving sends `Command(resume=ApprovalResponse)`; the resume value is validated against a Pydantic schema. The approval survives restarts and can come from another process (tested). Checkpoint deserialization uses an explicit type allowlist rather than loading arbitrary classes.

**Tool adapters behind one client.** The agent only knows `StoreClient` and the Pydantic schemas. Here the client talks HTTP to a bundled mock Store API (in-process by default, or remote via `STORE_API_URL`); in a real project the same methods call Shopify, Zendesk or a CRM. Tools return both compact JSON for the model and a typed artifact that becomes facts for the rules.

**Every step is audited.** Each node returns audit events through an append-only reducer in the graph state, mirrored into an `audit_events` table: LLM outputs, tool calls with arguments and results, each rule outcome, the verdict, the operator decision and the executed action. Support leads can see *why* a refund happened, engineers can debug a bad classification, and the trail doubles as a dataset for evaluations.

**Replies are grounded in an explicit context.** `draft_reply` receives a whitelisted context (resolution, order facts, refund reference, KB articles), not the whole state. Order details are withheld when the sender's identity is not confirmed, and operator notes never reach the customer (both tested). The prompt tells the model the decision is final and forbids promises the context does not contain.

**Evaluate what has ground truth exactly, judge only the rest.** Intent, verdict, resolution and the retrieved article are compared with labels, so those scores are exact and reproducible. Only reply quality goes to an LLM judge, which sees the same whitelisted context as the reply writer and must explain every score. A different model family judges than writes (GPT agent, Claude judge in the run below) to avoid self-preference.

**A fake model that implements the real interface.** `FakeSupportModel` subclasses LangChain's `BaseChatModel` and supports `bind_tools` and `with_structured_output`, so the whole graph (tool loop, interrupts, replies) runs in tests, CI and Docker without keys and with reproducible output.

## Testing

```bash
make test    # 216 tests; the 12 Postgres/pgvector ones are skipped unless TEST_POSTGRES_URL is set (CI sets it)
make lint    # ruff check, ruff format --check, mypy --strict
make eval    # 40 labelled tickets, fake agent + fake judge
```

| Suite | What it covers |
|---|---|
| `test_rules.py` | Every business rule with parametrized edge cases: exactly 30/31 days, $99.99/$100.00/$100.01, VIP on/off, identity (case-insensitive, mismatch, unverified), address statuses, KB score threshold, verdict severity ordering, config overrides |
| `test_store_and_tools.py` | Mock Store API endpoints and conflicts, KB ranking, HTTP client error mapping, tool content + artifact, error results |
| `test_fake_llm.py` | Entity extraction, classification of all examples, structured output and multi-step tool calling through the standard LangChain interface |
| `test_graph.py` | All 8 scenarios end to end; interrupt -> approve and interrupt -> reject; resume after restart from the checkpoint; double review rejected; VIP / unverified sender / identity mismatch / unknown order; store conflict during execution -> escalation; LLM looking up the wrong order; LLM failure -> ticket marked failed |
| `test_api.py` | REST endpoints, validation, 404/409, review queue filter, UI pages, form submit, htmx fragment and no-JS fallback |
| `test_cli.py` | `run`, `example`, pause + approve in a separate invocation, `demo`, `show`, `graph` |
| `test_knowledge.py` | Hashing embeddings, retrieval ranking and scores on the memory and pgvector backends, incremental pgvector sync (only changed articles re-embedded, table rebuilt on a dimension change), unrelated queries below the threshold |
| `test_evals.py` | Dataset integrity, deterministic judge findings, metrics, stratified sampling, usage/cost accounting, the eval CLI and its `--fail-under` gate |
| `test_providers.py` | Every `LLM_PROVIDER` / `EMBEDDINGS_PROVIDER` builds a correctly configured client (OpenRouter endpoint, key, headers, plain-text embeddings) |
| `test_tracing.py` | One trace per ticket run with node, LLM (GenAI attributes, token usage) and tool spans; interrupts recorded as events, LLM failures as errors; off by default; console exporter |
| `test_postgres.py` | Pause and resume with the Postgres checkpointer (runs in GitHub Actions CI with a pgvector Postgres service) |

## Adapting to your stack

The agent depends only on `StoreClient` ([`tools/client.py`](src/support_agent/tools/client.py)) and the schemas in [`store_api/schemas.py`](src/support_agent/store_api/schemas.py). Integrating with real systems means implementing those methods and wiring the inbound/outbound ends:

| Need | Typical system | What to implement |
|---|---|---|
| Orders, refunds, address changes | Shopify, WooCommerce, BigCommerce | `get_order`, `create_refund`, `update_shipping_address` against the Admin API |
| Customer profile and tier | HubSpot, Salesforce, Klaviyo | `get_customer` (map a VIP property or segment to `tier`) |
| Knowledge base | Zendesk Guide, Gorgias, Notion | `list_knowledge_articles` (the articles are embedded and indexed by the app; pgvector re-embeds only what changed) |
| Inbound tickets and replies | Zendesk, Gorgias, Freshdesk, Intercom | Webhook -> `POST /tickets`; post `reply` back as a draft or public reply |
| Escalation | Same helpdesk | `create_escalation` -> assign to a group, set priority, add a tag |

Business rules, prompts and the graph stay the same. Rules specific to your shop (e.g. "no refunds on sale items", "free replacement instead of refund for damaged goods") are added as new rule functions.

## Roadmap

Not implemented yet; natural next steps for a production rollout:

- Helpdesk connectors (Zendesk / Gorgias webhooks in, draft replies out) and a Shopify `StoreClient`.
- Operator authentication (SSO), roles, and approval notifications in Slack.
- Background workers for LLM calls; tickets are currently processed inside the HTTP request.
- Editable draft replies in the UI before sending; multilingual tickets.
- Hybrid retrieval (keyword + vector) and a reranker once the help center grows past a few hundred articles.

## License

[MIT](LICENSE) © 2026 Ivan Savchenko
