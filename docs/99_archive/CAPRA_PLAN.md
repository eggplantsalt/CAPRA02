# CAPRA Plan (Phase 0)

## Scope
This document records the Phase 0 repository survey and the minimal-intrusion CAPRA-v1 overlay plan on top of upstream openvla-oft.

## Upstream Entry Files Observed
Based on direct reads in this phase:
- Top-level usage and project entry docs: README.md, LIBERO.md
- LIBERO evaluation entry: experiments/robot/libero/run_libero_eval.py
- Evaluation/model utilities used by eval entry:
  - experiments/robot/openvla_utils.py
  - experiments/robot/robot_utils.py
- Training entry: vla-scripts/finetune.py

## Planned CAPRA Overlay Paths
Keep baseline behavior unchanged and place new CAPRA logic in dedicated overlay paths:
- experiments/robot/capra/
- scripts/capra/
- tests/capra/
- docs/ (CAPRA planning/context/progress docs)

## Mismatches vs Expected Layout
Observed during Phase 0:
- docs/CAPRA_CONTEXT.md did not exist before this phase.
- progress_capra.md did not exist before this phase.
- tests_capra.json did not exist before this phase.
- tests/ directory did not exist before this phase and is created for CAPRA-only tests.

No baseline code-path mismatch was found for the specified upstream entry scripts; paths from the phase prompt exist.

## Planned File Landing by Next Phases
Phase 1 (adapter/state access skeleton implementation only):
- experiments/robot/capra/env_adapter.py
- experiments/robot/capra/state_api.py
- tests/capra/test_env_adapter_roundtrip.py
- tests/capra/test_state_api_signals.py

Phase 2 (local evaluator v1):
- experiments/robot/capra/local_evaluator.py
- experiments/robot/capra/proposals.py
- experiments/robot/capra/footprint.py
- experiments/robot/capra/task_progress.py
- tests/capra/test_local_evaluator_v1.py

Phase 3 (mining v1 pipeline):
- experiments/robot/capra/mining.py
- experiments/robot/capra/supervision_io.py
- experiments/robot/capra/run_capra_mining.py
- scripts/capra/mine_capra_v1.sh
- tests/capra/test_mining_v1.py

Phase 4 (finetune integration as overlay entry):
- vla-scripts/finetune_capra.py
- experiments/robot/capra/training_targets.py
- tests/capra/test_finetune_capra_smoke.py

Phase 5 (metrics and evaluation wiring):
- experiments/robot/capra/metrics.py
- experiments/robot/capra/run_capra_eval.py
- scripts/capra/eval_capra_v1.sh
- tests/capra/test_metrics_v1.py

Phase 6 (thin benchmark adapters):
- experiments/robot/capra/benchmark_adapters.py
- docs/CAPRA_BENCHMARK_ADAPTERS.md
- tests/capra/test_benchmark_adapters.py

## Integration Guardrails
- Do not modify baseline behavior in:
  - experiments/robot/libero/run_libero_eval.py
  - experiments/robot/openvla_utils.py
  - experiments/robot/robot_utils.py
  - vla-scripts/finetune.py
- If unavoidable hook injection is needed later, record exact reason in progress_capra.md.
- No legacy shim paths should be introduced.
