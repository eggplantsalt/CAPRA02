# CAPRA Mining Realization (Production-Oriented)

## Scope and Goal

This document records how CAPRA mining has been upgraded from a runnable scaffold to a production-oriented data pipeline for paper experiments.

Primary goal:
- Produce supervision JSONL that is directly consumable by training-side strict lookup (sample key exact match).

Non-goal:
- Expand benchmark wrappers or add new benchmark abstractions.

## What Was Audited

The following modules were reviewed as requested:
- `experiments/robot/capra/core/local_evaluator.py`
- `experiments/robot/capra/core/mining.py`
- `experiments/robot/capra/core/proposals.py`
- `experiments/robot/capra/core/task_progress.py`
- `experiments/robot/capra/core/footprint.py`
- `experiments/robot/capra/pipelines/run_capra_mining.py`
- `scripts/capra/mine/mine_capra_v1.sh`

## Realization Status by Component

### 1. Local Evaluator (`local_evaluator.py`)

Current state:
- Uses deterministic weighted composition of novelty/progress/constraint/alignment.
- Includes temperature and top-k selection, and produces candidate statistics.

Production suitability:
- Suitable as a stable local scorer in a production mining loop.
- Hyperparameters are configurable via `MiningConfigV1` instead of hardcoded in the call path.

Known simplification:
- Scoring formulation is hand-crafted and static (not learned).

### 2. Mining Core (`mining.py`)

Current state:
- Runs step-level candidate generation and local evaluation.
- Writes strict-key supervision records (`sample_key`, `lookup_key`) aligned to training.
- Supports optional deferred label for non-hit timesteps.

Production suitability:
- Supervision schema and key construction are training-compatible.
- Output can be consumed by training loader without adapter glue.

Known simplification:
- Deferred behavior remains a policy choice; full closed-loop CAPRA scheduling is out of scope here.

### 3. Proposals (`proposals.py`)

Current state:
- Candidate proposals are generated around base action chunk using prefix-local templates.
- `prefix_steps` controls how many leading timesteps in the chunk can be modified; suffix steps remain base actions.
- `include_base` controls whether the identity/base proposal is kept as the local comparison anchor.
- Main templates include `speed_prefix_*`, `lift_prefix`, `retract_prefix`, `lift_then_go`, and optional lateral prefix offsets.
- Gripper/protected dimensions (`gripper_index`, `protected_dims`) are preserved by default and excluded from template scaling/noise.
- Gaussian perturbations are optional secondary candidates (`num_gaussian`, default off).
- Whole-chunk templates are secondary and disabled by default (`enable_whole_chunk_templates=False`).

Production suitability:
- Deterministic enough for reproducible mining with controlled random seed.

Known simplification:
- Proposal policy is heuristic, not yet learned from offline optimization.

### 4. Task Progress (`task_progress.py`)

Current state:
- Computes progress and constraints from adapter state and task metadata.

Production suitability:
- Works as a pluggable feature extractor for multiple env adapters.

Known simplification:
- Feature set is generic and conservative; task-specific richer progress features may improve quality.

### 5. Footprint (`footprint.py`)

Current state:
- Computes/updates trajectory footprint and novelty-relevant signals.

Production suitability:
- Supports novelty component used by local evaluator in production mining.

Known simplification:
- Footprint metric remains lightweight and intentionally simple for stability.

## Critical Pipeline Realization: Default Path Is Now Production

## Before

`run_capra_mining.py` had a demo-first default path and could run without real rollout data or a real environment factory.

## After

`experiments/robot/capra/pipelines/run_capra_mining.py` now requires production inputs by default:
- `--episodes_path`: rollout episodes JSONL
- `--env_factory`: environment factory spec in `module:function` form

The pipeline now:
1. Loads real episodes from JSONL.
2. Resolves environment factory from import path.
3. Creates env per episode metadata.
4. Mines supervision records via `mine_episode_v1`.
5. Writes training-consumable supervision JSONL.

Result:
- The default CLI path is production-oriented, not demo-oriented.

## Debug Path Isolation (Retained, Not Mixed)

Debug artifacts are retained but separated from training artifacts:
- Main output: supervision JSONL (training artifact)
- Optional debug output: `--debug_summary_path` JSON (diagnostic artifact)

This keeps diagnostics available without polluting or redefining the training data contract.

## Shell Entry Realization

`scripts/capra/mine/mine_capra_v1.sh` was updated to enforce production inputs:
- Required args: `<episodes_path> <env_factory> [output_path]`
- Missing required args now fail with usage text.

This prevents accidental execution of a demo path as if it were production.

## End-to-End Validation (Mining -> Training Hit)

Added:
- `tests/capra/test_mining_training_e2e.py`

Covered behavior:
- Mining output JSONL is generated from a production-like env/episode flow.
- Training-side supervision index loads the JSONL.
- Batch lookup hits for matching sample keys and does not trigger for non-matching keys.

## Deferred / Not Fully CAPRA Yet

The following are explicitly deferred and are not claimed as fully realized CAPRA:
- Learned proposal policy replacing heuristic proposal generation.
- Global multi-episode optimization and closed-loop retraining scheduler.
- Rich task-specific progress instrumentation beyond the current generic feature API.
- Benchmark-level expansion/wrapper refactor (intentionally out of scope).

## Why This Is Sufficient for Main Experiments

For paper-facing training experiments, the key requirement is reliable and reproducible production of supervision records that training can consume without schema/key drift. This realization satisfies that requirement by:
- Enforcing production inputs at entry points.
- Preserving strict key compatibility with training lookup.
- Providing explicit debug isolation.
- Backing the integration with end-to-end hit tests.

In short: this is a production-oriented CAPRA mining pipeline suitable for main experiment data generation, while clearly separating deferred full-CAPRA research extensions.
