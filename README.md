---
title: LatentGoalOps
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - benchmark
  - agents
---

# LatentGoalOps

LatentGoalOps is a research-grade OpenEnv benchmark for hidden-objective inference in realistic startup operations. Instead of telling an agent what to optimize, the environment gives indirect clues through KPI dashboards, stakeholder messages, backlog options, and event noise. The agent must infer the business objective, choose coherent actions, and in the hard task recover from a silent mid-episode goal shift.

This repo is designed for the OpenEnv hackathon and for a later paper submission. It implements:

- Full OpenEnv-style `reset()`, `step()`, and `state()` support
- Typed Pydantic `Action`, `Observation`, and `State` models
- Five tasks with deterministic programmatic graders
- Leakage-safe shaped rewards with partial progress signals
- A resumable baseline harness with OpenAI-client calls against OpenAI-compatible endpoints such as DigitalOcean Gradient
- A root [`inference.py`](/Users/avichaldwivedi/dev/latent-neurips/inference.py) submission runner for the required hackathon baseline flow
- Logging, checkpointing, aggregation, statistics, and plotting utilities
- A synthetic LLM operator baseline that behaves like a bounded startup leader rather than a score-maximizing agent
- Hugging Face Space metadata and Docker packaging

## Why This Is Real-World

The benchmark models tasks that real startup operators and product teams actually do:

- Task 1: triage customer feedback and escalations
- Task 2: prioritize a sprint roadmap under budget
- Task 3: run a startup for a week under noisy and shifting business constraints
- Task 4: allocate capital across nonlinear programs with diminishing returns
- Task 5: assemble a crisis-response package from initiatives plus operating policies

This is not a toy game. The hidden goal changes how the same visible evidence should be interpreted, which makes the benchmark useful for agent evaluation rather than just schema following.

The latest version also adds a persistent synthetic startup world with:

- visible customer accounts, including ACV, renewal windows, support tier, and relationship health
- recurring stakeholder personas with distinct agenda biases
- team-state signals such as capacity, burnout, and execution reliability
- market context and governance constraints that make some actions strategically unsafe

## Hackathon Submission Profile

The environment ships with 5 tasks, but the submission baseline intentionally defaults to the 3 core tasks required by the hackathon:

- `task1_feedback_triage`
- `task2_roadmap_priority`
- `task3_startup_week`

This keeps the mandatory [`inference.py`](/Users/avichaldwivedi/dev/latent-neurips/inference.py) runtime comfortably inside the submission budget while preserving the easy / medium / hard progression. Tasks 4 and 5 remain available in the environment for fuller benchmark runs.

## Environment API

The public OpenEnv surface is:

- `reset(seed: int | None = None, task_id: str | None = None) -> LatentGoalOpsObservation`
- `step(action: LatentGoalOpsAction) -> LatentGoalOpsObservation`
- `state -> LatentGoalOpsState`

The root manifest is in [`openenv.yaml`](/Users/avichaldwivedi/dev/latent-neurips/openenv.yaml), and the FastAPI entrypoint is in [`server/app.py`](/Users/avichaldwivedi/dev/latent-neurips/server/app.py).

## Observation Space

The unified observation includes these typed fields:

- `task_id`
- `step_index`
- `horizon`
- `sim_date`
- `sim_day_label`
- `task_summary`
- `narrative`
- `dashboard`
- `inbox`
- `backlog`
- `accounts`
- `stakeholders`
- `teams`
- `market_context`
- `governance_constraints`
- `alerts`
- `calendar_events`
- `decision_ledger`
- `pending_effects`
- `realized_effects`
- `budget_remaining`
- `capacity_remaining`
- `sprint_budget`
- `stakeholder_notes`
- `available_actions`
- `done`
- `reward`
- `metadata`

The dashboard contains realistic metrics such as `dau`, `mrr`, `d30_retention`, `ops_margin`, and `support_ticket_volume`.

## Action Space

The unified action model contains task-specific fields:

- Task 1: `labels`, `priorities`, `escalate_ids`
- Task 2: `selected_item_ids`, `rationale_summary`
- Task 3: `chosen_initiatives`, `messaging_action`, `pricing_change_pct`, `support_policy`, `rationale`
- Task 4: `budget_allocations`, `rationale_summary`
- Task 5: `chosen_initiatives`, `messaging_action`, `pricing_change_pct`, `support_policy`, `rationale`

Validation forbids irrelevant cross-task fields, so agents cannot submit malformed hybrid actions without penalty.

## Tasks

### Task 1: Weighted Feedback Triage

Difficulty: easy

The agent receives 8-12 customer messages and must:

- label each item correctly
- assign urgency from 1-5
- escalate up to 3 items

Each message is now tied to a visible customer account profile, so the agent must reason over content plus economics: ACV, renewal timing, support expectations, and relationship health. The hidden goal changes the category weighting of errors, churn risk, revenue issues, and efficiency problems, but it does not change the ground-truth label.

Grader:

- `0.50 * label_accuracy`
- `0.30 * priority_alignment`
- `0.20 * escalation_overlap`

### Task 2: Roadmap Prioritization Under Budget

Difficulty: medium

The agent receives a visible backlog with costs, KPI deltas, beneficiary segments, linked accounts, implementation risk, and policy tags. Recurring stakeholder notes only indirectly reveal the true objective. The hidden goal controls which KPI improvements matter most.

Grader:

- normalized hidden utility against a reproducible random baseline
- budget waste penalty for leaving too much unused capacity

### Task 3: Startup Week

Difficulty: hard

The agent operates a startup over a dated operating calendar. Each day includes a narrative briefing, inbox messages, a backlog, alerts, explicit upcoming calendar events, a decision ledger from prior days, delayed effects still in flight, customer accounts with renewal windows and contract value, internal team state, market context, and visible governance constraints. In some episodes the hidden goal silently shifts in the middle, forcing the agent to adapt without being told.

Grader:

- terminal latent utility
- adaptation score after goal shift
- trajectory coherence
- constraint adherence

### Task 4: Capital Allocation Under Uncertainty

Difficulty: medium

The agent receives a visible menu of operating programs and must allocate discrete budget points across them. Programs expose visible allocation caps, saturation hints, beneficiary segments, dependencies, conflicts, and implementation risk. The hidden goal determines which returns matter most, but the visible structure makes over-allocation and scattered portfolios strategically costly.

Grader:

- normalized hidden utility against a reproducible random-allocation baseline
- budget use quality under a visible budget cap

### Task 5: Crisis Response Package

Difficulty: hard

The agent must choose one executive response package under pressure: a small set of initiatives plus pricing, messaging, and support policy. The observation includes a crisis narrative, inbox pressure, high-touch accounts, governance constraints, and resource limits. The challenge is to infer the latent objective while still avoiding obviously unsafe policy combinations.

Grader:

- normalized package value against a reproducible random baseline
- explicit constraint adherence

## Reward Design

The reward is intentionally leakage-safe. It only depends on observable KPI movement and action coherence, not on the hidden weight vector directly.

Step reward combines:

- KPI improvement reward
- strategy coherence reward
- invalid action penalty
- budget waste penalty
- KPI damage penalty
- governance-violation penalty

This gives useful partial progress feedback without letting the agent recover the latent objective by probing the reward surface.

Task 3 now also exposes simulation-time structure for richer research analysis:

- dated observations via `sim_date`
- explicit `decision_ledger`
- `pending_effects` and `realized_effects`
- visible `calendar_events` that let agents reason about delayed consequences
- account-level renewals and contract-risk signals that land later in the episode
- market and team state that can worsen when the wrong strategy keeps compounding

## Persistent Entity World

Every episode is built inside a synthetic but stateful startup world rather than a one-off prompt bundle.

The world currently contains:

- `CustomerAccount` entities with segment, seat count, ACV, renewal timing, support tier, security sensitivity, churn propensity, and strategic importance
- `StakeholderPersona` entities for roles such as CEO, CFO, CTO, Head of CS, and Growth Lead
- `InternalTeamState` entities that expose capacity, burnout risk, reliability, specialization, and cross-team friction
- `MarketContext` state such as runway, board pressure, compliance exposure, and pipeline health
- `GovernanceConstraint` rules for pricing, SLA handling, and margin preservation

Task 3 mutates this world over time:

- pricing and support actions can trigger governance failures
- initiatives can target specific accounts and land with delayed effects
- renewal windows tick down each simulated day
- renewals and churn scares create dated business outcomes
- team state and board pressure evolve as consequences accumulate

The more detailed design note for this layer lives in [paper/ENTITY_WORLD_SPEC.md](/Users/avichaldwivedi/dev/latent-neurips/paper/ENTITY_WORLD_SPEC.md).

## Credentials

The submission baseline uses the required hackathon env vars and the official OpenAI Python client.

Recommended env vars:

```env
API_BASE_URL=https://inference.do-ai.run/v1
MODEL_NAME=openai-gpt-oss-20b
HF_TOKEN=...
WANDB_API_KEY=...
```

Notes:

- `HF_TOKEN` is the required submission variable name for the baseline script. If you are using DigitalOcean Gradient, point `API_BASE_URL` at the Gradient endpoint and provide that token through `HF_TOKEN`.
- `DIGITALOCEAN_API_TOKEN` and `OPENAI_BASE_URL` are still accepted as backward-compatible fallbacks for local runs.
- `WANDB_API_KEY` is optional unless you wire W&B logging into your runs.

## Token Exhaustion and Recovery

This repo treats provider failure as three separate problems:

- invalid or revoked credential
- temporary rate or capacity limit
- account or budget exhaustion

Implemented behavior:

- `401/403`: fail fast, persist checkpoint, allow retry with another configured credential source later
- `429/5xx`: exponential backoff with jitter
- budget exhaustion: checkpoint current progress and stop cleanly

The baseline harness logs token usage and estimated spend per step, so long sweeps can resume instead of restarting from scratch. The full write-up is in [paper/TOKEN_EXHAUSTION_AND_RECOVERY.md](/Users/avichaldwivedi/dev/latent-neurips/paper/TOKEN_EXHAUSTION_AND_RECOVERY.md).

## Progress and Time Logging

Yes, the runner now logs timing so you can track whether the sweep is progressing normally.

Per-step and per-episode logs include:

- `started_at`
- `finished_at`
- `elapsed_seconds`
- episode-level `progress_fraction`
- episode-level `eta_seconds`
- run-level cumulative token and cost totals

These show up in both [`outputs/.../runs.jsonl`](/Users/avichaldwivedi/dev/latent-neurips/outputs) and W&B when `--wandb` is enabled.

## Local Setup

### 1. Create and sync the `uv` environment

```bash
uv sync
```

### 2. Run tests

```bash
uv run pytest
```

### 3. Start the server locally

```bash
uv run server --port 8000
```

### 4. Validate the environment structure

```bash
uv run python -m openenv.cli validate .
```

## Baselines

### Submission baseline

This is the required hackathon entrypoint. It uses the OpenAI Python client against an OpenAI-compatible endpoint and defaults to the 3 core submission tasks.

```bash
export API_BASE_URL=https://inference.do-ai.run/v1
export MODEL_NAME=openai-gpt-oss-20b
export HF_TOKEN=...
uv run python inference.py
```

Optional local Gradient compatibility:

```bash
set -a
source .env
export API_BASE_URL="${API_BASE_URL:-https://inference.do-ai.run/v1}"
export MODEL_NAME="${MODEL_NAME:-openai-gpt-oss-20b}"
export HF_TOKEN="${HF_TOKEN:-$DIGITALOCEAN_API_TOKEN}"
set +a
uv run python inference.py --tasks task1_feedback_triage,task2_roadmap_priority,task3_startup_week
```

### Heuristic baseline

```bash
uv run baseline --policy heuristic --tasks all --seeds 100:3
```

### Random baseline

```bash
uv run baseline --policy random --tasks all --seeds 100:3
```

### Oracle sanity baseline

```bash
uv run baseline --policy oracle --tasks all --seeds 100:3
```

### Gradient model baseline

```bash
uv run baseline --policy model --model openai-gpt-oss-20b --tasks all --seeds 100:1
```

### Synthetic operator baseline

```bash
uv run baseline --policy synthetic_operator --persona-model openai-gpt-oss-20b --operator-style auto --tasks all --seeds 100:3
```

Synthetic operator mode is intentionally different from the score-maximizing model baseline. It is best treated as a fixed human-proxy baseline: one persona-source model is held constant, while the benchmarked agent models vary separately.

### Gradient model baseline with W&B logging

```bash
set -a
source .env
set +a
uv run baseline --policy model --model openai-gpt-oss-20b --tasks all --seeds 100:10 --output-dir outputs/wandb-gptoss20b --run-id gptoss20b-val --wandb --wandb-project latentgoalops --wandb-group validation
```

### Optional persona-source sensitivity check

```bash
uv run sweep-models --policy synthetic_operator --models openai-gpt-oss-20b,openai-gpt-oss-120b --operator-style auto --tasks all --seeds 100:10 --output-root outputs/operator-ladder
```

The default synthetic-operator workflow does not need a sweep. The command above is only for appendix-style sensitivity checks where the persona-source model itself is varied. In `synthetic_operator` mode, an explicit `--models` list is interpreted as the persona-source models to compare.

### Run the full benchmark model ladder through `gpt-oss-120b`

```bash
set -a
source .env
set +a
uv run sweep-models --tasks all --seeds 100:10 --output-root outputs/model-ladder --wandb --wandb-project latentgoalops --wandb-group model-ladder
```

Default ladder:

- `openai-gpt-oss-20b`
- `openai-gpt-oss-120b`

### Recommended research flow

1. Run one fixed synthetic-operator baseline with `openai-gpt-oss-20b` as the persona-source model.

```bash
uv run baseline --policy synthetic_operator --persona-model openai-gpt-oss-20b --operator-style auto --tasks all --seeds 100:10 --output-dir outputs/persona-baseline/core
uv run baseline --policy synthetic_operator --persona-model openai-gpt-oss-20b --operator-style auto --tasks all --seeds 100:10 --scenario-split heldout --output-dir outputs/persona-baseline/heldout
```

2. Run each benchmarked agent model one by one into its own directory tree.

```bash
uv run baseline --policy model --model openai-gpt-oss-20b --tasks all --seeds 100:10 --output-dir outputs/main-benchmark/openai-gpt-oss-20b/core
uv run baseline --policy model --model openai-gpt-oss-20b --tasks all --seeds 100:10 --scenario-split heldout --output-dir outputs/main-benchmark/openai-gpt-oss-20b/heldout
uv run baseline --policy model --model openai-gpt-oss-120b --tasks all --seeds 100:10 --output-dir outputs/main-benchmark/openai-gpt-oss-120b/core
uv run baseline --policy model --model openai-gpt-oss-120b --tasks all --seeds 100:10 --scenario-split heldout --output-dir outputs/main-benchmark/openai-gpt-oss-120b/heldout
```

3. Aggregate later once enough runs have accumulated. The reporting stack recursively scans nested `runs.jsonl` files, so you can benchmark models one at a time now and club them together at the end.

```bash
uv run plots --input outputs/main-benchmark --output-dir outputs/main-benchmark-plots
uv run plots --input outputs/persona-baseline --output-dir outputs/persona-baseline-plots
```

### Reliability evaluation (`pass^k` / repeated runs)

```bash
uv run reliability \
  --models openai-gpt-oss-20b,openai-gpt-oss-120b \
  --tasks all \
  --seeds 100:10 \
  --repeats 5 \
  --temperature 0.2 \
  --output-root outputs/reliability
```

### Core ablation suite

```bash
uv run ablations \
  --models openai-gpt-oss-20b,openai-gpt-oss-120b \
  --tasks all \
  --seeds 100:10 \
  --output-root outputs/ablations
```

Implemented ablations:

- `full`
- `no_shift`
- `no_delay`
- `no_ledger`
- `sparse_reward`

### Held-out scenario split

```bash
uv run baseline --policy heuristic --tasks all --seeds 100:3 --scenario-split heldout
```

The held-out split swaps in unseen initiative and event families for the strategic tasks. This is intended as a light OOD check against benchmark-specific memorization.

### Leakage audit

```bash
uv run leakage-audit --tasks all --seeds 100:40 --output outputs/leakage_audit.json
```

This audits whether the initial observation is still too predictive of the hidden goal and flags explicit goal-name leakage phrases.

### Human baseline collection

Collect a human trace:

```bash
uv run human-collect --participant avichal --task task3_startup_week --seed 100
```

The collector now supports an interactive menu-driven flow for humans by default. Use `--raw-json` only if you want to paste actions manually.

Score collected traces:

```bash
uv run human-score --input outputs/human --output outputs/human/human_scores.json
```

### Trajectory audit packets

```bash
uv run audit-packets --input outputs --output-dir outputs/audit_packets
```

This creates top / median / bottom trajectory packets per model-task pair for manual review.

## Expected Baseline Behavior

The exact numbers depend on seeds, but the intended score ordering is:

- oracle > heuristic > random
- Task 3 should remain the hardest temporal task
- Task 4 and Task 5 should be meaningfully harder than Task 1 and usually harder than Task 2
- the hard temporal task should show visible degradation under goal shifts

### Verified Smoke Baselines

Fresh local smoke summaries for the 5-task benchmark now live in:

- [`outputs/random-smoke-v3/summary.json`](/Users/avichaldwivedi/dev/latent-neurips/outputs/random-smoke-v3/summary.json)
- [`outputs/heuristic-smoke-v3/summary.json`](/Users/avichaldwivedi/dev/latent-neurips/outputs/heuristic-smoke-v3/summary.json)
- [`outputs/oracle-smoke-v3/summary.json`](/Users/avichaldwivedi/dev/latent-neurips/outputs/oracle-smoke-v3/summary.json)

These are sanity checks, not paper numbers. The exact scores move with seeds, but the important contract is still oracle > heuristic > random and a visible difficulty gap between the easy triage task and the harder strategic tasks.

### Verified Submission Smoke Baseline

A local model-backed smoke run of the required [`inference.py`](/Users/avichaldwivedi/dev/latent-neurips/inference.py) path now lives in [`outputs/submission-final-smoke-v2/summary.json`](/Users/avichaldwivedi/dev/latent-neurips/outputs/submission-final-smoke-v2/summary.json).

Mean scores for `openai-gpt-oss-20b` with `seeds=100:1` on the 3 core submission tasks:

- `task1_feedback_triage`: `0.8513`
- `task2_roadmap_priority`: `0.4885`
- `task3_startup_week`: `0.6662`

This smoke run completed in about 40 seconds locally and triggered two parse fallbacks across the 3-task sweep, which are already recorded in the summary artifact.

## Outputs

Baseline runs write:

- `outputs/baseline/runs.jsonl`
- `outputs/baseline/checkpoint.json`
- `outputs/baseline/summary.json`
- `outputs/submission-baseline/runs.jsonl`
- `outputs/submission-baseline/checkpoint.json`
- `outputs/submission-baseline/summary.json`

`summary.json` now separates:

- overall mean scores
- strict mean scores without parse rescue
- rescued mean scores when repair / fallback logic was needed

These logs are consumed by the analysis helpers under [`src/latentgoalops/analysis`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/analysis).

## Plot Generation

Generate a plot bundle from one run directory or the entire `outputs/` tree:

```bash
uv run plots --input outputs --output-dir outputs/plots
```

The plot bundle also writes `summary_tables.json` with score means, confidence intervals, cost, and runtime.

Current plot bundle includes:

- grouped task score bars
- score distribution box plots
- score heatmap
- cost vs score scatter
- reward trajectory curves
- token usage bars
- parse fallback rates

## W&B Login

Interactive login:

```bash
set -a
source .env
set +a
uv run wandb login
```

Non-interactive login:

```bash
set -a
source .env
set +a
uv run wandb login "$WANDB_API_KEY"
```

## Docker and Hugging Face Space

This repo includes a root [`Dockerfile`](/Users/avichaldwivedi/dev/latent-neurips/Dockerfile) for Docker-based HF Spaces. Since Docker is not installed locally in this workspace, development is done via `uv run ...`, but the container target is still authored for submission.

Suggested HF runtime secrets / variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If you deploy with DigitalOcean Gradient as the backing model endpoint, set:

- `API_BASE_URL=https://inference.do-ai.run/v1`
- `MODEL_NAME=<your Gradient model id>`
- `HF_TOKEN=<your Gradient token>`

## Repository Layout

- [`src/latentgoalops/models.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/models.py): typed public models
- [`src/latentgoalops/server/environment.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/server/environment.py): main environment
- [`src/latentgoalops/server/tasks`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/server/tasks): task generators and dynamics
- [`src/latentgoalops/server/grader.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/server/grader.py): deterministic graders
- [`src/latentgoalops/baseline/run_baseline.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/baseline/run_baseline.py): baseline runner
- [`src/latentgoalops/logging_`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/logging_): JSONL logging schemas and writer
- [`src/latentgoalops/analysis`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/analysis): aggregation, stats, and plotting
