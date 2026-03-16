# CAPRA Benchmark Adapters (Phase 6)

## Goal
Provide a thin benchmark adapter layer for CAPRA without rewriting upstream LIBERO evaluation.

## Preserved Upstream Utility Path
Use the unchanged upstream utility evaluation entry:
- `python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint <CKPT> --task_suite_name <SUITE>`

This remains the standard LIBERO utility path and is not wrapped by a new framework.

## SafeLIBERO Adapter
CAPRA Phase 6 adapter lives in:
- `experiments/robot/capra/benchmark_adapters.py`

Smoke adapter function:
- `smoke_run_safelibero(SafeLiberoSmokeConfig(...))`

Expected local external repo path:
- `vlsa-aegis/safelibero`

### Install Notes (External Dependency)
From repository root:

```bash
cd vlsa-aegis/safelibero
pip install -r requirements.txt
pip install -e .
```

This follows the external SafeLIBERO project setup and keeps CAPRA integration thin.

## Custom Split Adapter (Thin)
Current supported high-value splits only:
- `support-critical-neighbor`
- `chain-reaction`

Smoke adapter function:
- `smoke_run_custom_split(split_name)`

No precursor/SAB/full dataset builder logic is introduced in this phase.

## CAPRA Eval Runner Integration
Extended runner:
- `experiments/robot/capra/run_capra_eval.py`

Example smoke runs:

```bash
# SafeLIBERO adapter smoke
python -m experiments.robot.capra.run_capra_eval \
  --benchmark_mode safelibero \
  --safelibero_root vlsa-aegis/safelibero \
  --task_suite_name safelibero_spatial \
  --safety_level I

# Custom split adapter smoke
python -m experiments.robot.capra.run_capra_eval \
  --benchmark_mode custom_split \
  --custom_split support-critical-neighbor
```

## Shell Entrypoint
- `scripts/capra/eval_capra_v1.sh`

Defaults to tiny metric smoke mode, and supports benchmark adapter modes via optional args.
