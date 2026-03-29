# Synthetic Operator Baseline

LatentGoalOps now includes a synthetic operator baseline that is meant to model a plausible startup operator rather than a benchmark-optimizing agent.

## Why It Exists

The benchmark paper should separate at least two policy families:

- score-maximizing agent policies
- realistic operator-style policies

This helps answer whether the environment rewards brittle optimization or robust operational judgment.

## What It Is

The synthetic operator uses the same model API path as the normal model baseline, but changes two things:

1. Prompting

- the system prompt frames the model as a bounded operator
- the prompt includes a reproducible persona
- the model is told to prefer defendable decisions, lower-regret moves, and realistic operational discipline

2. Guardrails

- limits the number of parallel bets
- limits the number of simultaneous startup initiatives
- clamps extreme pricing changes
- avoids automation-first support for premium / strategic renewal-risk accounts when visible guardrails suggest that would be unrealistic

## Personas

Current operator personas:

- `founder`
- `product`
- `support`
- `finance`
- `gm`

`auto` mode chooses a deterministic persona per `(seed, task)` so runs stay reproducible.

## Recommended Paper Framing

Use this as a distinct approach, not as a human baseline.

Suggested language:

- "synthetic operator baseline"
- "LLM-simulated operator policy"
- "operator-style policy"

Avoid:

- "human baseline"
- "human-equivalent policy"

## Example Question It Helps Answer

- Does a score-maximizing agent outperform a realistic bounded operator?
- Does the benchmark reward focused operational judgment or aggressive benchmark gaming?
- Which tasks benefit from optimization pressure versus conservative real-world discipline?
