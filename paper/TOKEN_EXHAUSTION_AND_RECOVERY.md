# Token Exhaustion and Recovery Addendum

This addendum extends the LatentGoalOps technical blueprint with a concrete provider-failure policy for long-running benchmark sweeps.

## Why This Matters

Research runs over many seeds and multiple models fail in ways that are not semantically equivalent:

- invalid or revoked token
- temporary service saturation
- hard account or spending exhaustion

Treating all of these as a generic “API error” destroys reproducibility because the run either loops forever or silently drops episodes.

## Failure Taxonomy

### 1. Credential Failure

Examples:

- `401 Unauthorized`
- `403 Forbidden`

Interpretation:

- token revoked
- token missing required provider permission
- provider-side auth behavior changed

Recovery:

- stop immediately
- persist checkpoint with `failure_mode=ProviderCredentialError`
- require operator intervention or fallback token source

### 2. Temporary Capacity Failure

Examples:

- `429 Too Many Requests`
- transient `5xx`
- connection resets or upstream timeouts

Interpretation:

- rate limiting
- provider overload
- temporary transport issue

Recovery:

- retry with exponential backoff and jitter
- preserve step-level token accounting
- only fail terminally after bounded retries

### 3. Budget / Quota Exhaustion

Examples:

- provider reports billing, quota, or credit exhaustion
- local experiment spend exceeds configured budget cap

Recovery:

- write checkpoint immediately
- stop cleanly
- do not continue with degraded partial logging

## Code Path

The baseline provider adapter in [`src/latentgoalops/baseline/providers.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/baseline/providers.py) implements:

- deterministic retry behavior
- credential failure detection
- budget-exhaustion exceptions
- token and estimated cost accounting

The batch runner in [`src/latentgoalops/baseline/run_baseline.py`](/Users/avichaldwivedi/dev/latent-neurips/src/latentgoalops/baseline/run_baseline.py) implements:

- resumable checkpoint state
- cumulative token usage
- cumulative spend tracking
- fail-fast behavior for fatal provider errors

## Checkpoint Contents

Each checkpoint stores:

- run id
- model name
- task list
- seed list
- next pending job index
- cumulative input tokens
- cumulative output tokens
- cumulative estimated spend
- failure mode

This is enough to resume a partial sweep without rerunning the successful prefix.

## Recommended Fallback Ladder

When cost pressure matters more than exact model identity, use a descending-cost ladder such as:

- `deepseek-r1-distill-llama-70b`
- `alibaba-qwen3-32b`
- `openai-gpt-oss-120b`
- `openai-gpt-oss-20b`

This should be a conscious experiment policy choice, not an automatic silent substitution.

## Reporting Recommendation

Paper tables and appendix notes should distinguish:

- completed runs
- resumed runs
- interrupted runs due to provider failure

This preserves transparency and prevents hidden survivorship bias in benchmark reporting.

