# CAPRA Progress Log

## Phase 0 - Repository Survey and Scaffold
Status: completed

Completed in this phase:
- Verified repository root and git status before edits.
- Read required baseline files:
  - README.md
  - LIBERO.md
  - experiments/robot/libero/run_libero_eval.py
  - experiments/robot/openvla_utils.py
  - experiments/robot/robot_utils.py
  - vla-scripts/finetune.py
- Created CAPRA documentation scaffold:
  - docs/CAPRA_PLAN.md
  - docs/CAPRA_CONTEXT.md
- Created CAPRA code/test/script scaffold:
  - experiments/robot/capra/
  - scripts/capra/
  - tests/capra/
- Added placeholder CAPRA module files (no business logic).
- Added import smoke test for CAPRA package.
- Executed required minimal test: `pytest tests/capra/test_imports.py -q`.

Baseline safety check:
- No baseline file was modified in this phase.

Remaining risks/blockers:
- tests/ did not exist previously; this is now created for CAPRA tests.
- LIBERO/ appears as an untracked directory in git status; treat as external/submodule-like content and do not modify as part of CAPRA overlay.

Test notes:
- First run failed due to missing repo root on `PYTHONPATH` under pytest.
- Fixed by adding repository-root path insertion inside tests/capra/test_imports.py.
- Final result: 2 passed.

Next step:
- Phase 1: env adapter and state_api round-trip scaffolding with tests, still overlay-only.

## Phase 1 - env_adapter + state_api Infrastructure
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/env_adapter.py:
  - Unified EnvAdapter wrapper.
  - get_sim() as sanctioned simulator access path.
  - snapshot/capture/restore APIs with stable state round-trip path.
- Implemented experiments/robot/capra/state_api.py:
  - read_state_signals(env_adapter, obs, info).
  - obs-first read for pose/joint/gripper fields.
  - sim-backed read for dynamics/contact fields (qvel/contact count when available).
- Added data structures in experiments/robot/capra/types.py:
  - EnvSnapshot
  - StateSignals
- Added tests:
  - tests/capra/test_env_adapter.py
  - tests/capra/test_state_api.py
- Updated package exports in experiments/robot/capra/__init__.py.

Tests run:
- pytest tests/capra -q
- Result: 5 passed.

Current limitations (explicit):
- Exact simulator snapshot/restore semantics can vary by backend/environment wrapper.
- Current restore logic uses the most stable available path in order:
  sim.set_state_from_flattened -> sim.set_state -> env.set_state -> env.set_init_state.
- This is sufficient for round-trip scaffolding tests but should be validated on real LIBERO runtime in a later phase.

Baseline safety check:
- No baseline upstream file behavior was changed.

Next step:
- Phase 2: local evaluator v1 scaffolding (short rollout + progress-preserving gate + footprint v1), still overlay-only.

## Phase 2 - Local Candidate Evaluator Core
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/proposals.py:
  - Finite local proposal builder around base action chunk.
  - Includes base, speed-scaled variants, small lift, small retract.
  - Optional Gaussian local perturbations with configurable count and std.
  - Proposal count kept small and bounded by max_proposals.
- Implemented experiments/robot/capra/task_progress.py:
  - compute_progress_features_v1(...)
  - equivalent_progress_gate(...)
  - Progress-preserving filtering only (no full V_prog).
- Implemented experiments/robot/capra/footprint.py:
  - compute_footprint_v1(...)
  - Outputs displacement_total, severe_event_flags, severe_penalty, components.
  - v1 default uses displacement + severe events only.
- Implemented experiments/robot/capra/local_evaluator.py:
  - evaluate_candidate_v1(...)
  - evaluate_candidates_v1(...)
  - summarise_candidate_results(...)
  - Uses same snapshot for short counterfactual rollout of each candidate.
- Extended experiments/robot/capra/types.py with Phase 2 dataclasses:
  - ActionProposal, ProgressFeaturesV1, FootprintV1, CandidateEvalV1, CandidateSummaryV1.
- Extended experiments/robot/capra/state_api.py to carry object_positions/object_velocities from obs/info.
- Added tests:
  - tests/capra/test_task_progress_v1.py
  - tests/capra/test_footprint_v1.py
  - tests/capra/test_local_evaluator_v1.py

Tests run:
- pytest tests/capra -q
- Result: 9 passed.

Constraints check:
- No contact impulse implementation.
- No exact support-break modeling implementation.
- No precursor attribution implementation.
- No buffer introduced.
- No baseline upstream behavior modified.

Remaining risks/blockers:
- Progress gate in v1 is intentionally approximate and relies on available target-distance/info fields when provided.
- Real LIBERO runtime may require additional task-specific adapters for richer severe-event signals.

Next step:
- Phase 3: mining v1 pipeline from base rollouts to safer supervision tuples.

## Phase 3 - Supervision Mining
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/mining.py:
  - mine_one_timestep_v1(...)
  - mine_episode_v1(...)
  - MiningConfigV1 for epsilon_p, delta_min, w_max, short_horizon and proposal config.
- Implemented experiments/robot/capra/supervision_io.py:
  - SupervisionRecord schema dataclass.
  - write_supervision_jsonl(...)
  - read_supervision_jsonl(...)
- Implemented experiments/robot/capra/run_capra_mining.py:
  - run_capra_mining(...) for small episode lists.
  - Minimal demo CLI path that writes mined supervision JSONL.
- Added script entrypoint:
  - scripts/capra/mine_capra_v1.sh
- Added tests:
  - tests/capra/test_mining_v1.py

Tests run:
- pytest tests/capra -q
- Result: 11 passed.

Phase smoke execution:
- python -m experiments.robot.capra.run_capra_mining --output_path tmp/capra/mined_v1.jsonl
- Output: Wrote 1 records (non-empty under safer-candidate condition).

Constraints check:
- No SAB implementation.
- No merged dataset builder implementation.
- No complex resume/checkpoint framework added.
- No precursor attribution implementation.
- No full eval implementation.

Remaining risks/blockers:
- Current runner uses a tiny demo env path for smoke execution; real benchmark rollout integration remains a later phase task.
- Record stores minimal observation input or reference payload; final training adapter may require additional normalization/packing.

Next step:
- Phase 4: finetune integration with anchor + weighted safer-target regression.

## Phase 4 - finetune_capra Training Wiring
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/training_targets.py:
  - supervision_record_to_target(...)
  - collate_training_targets(...)
  - Converts mined supervision records into batched training tensors.
- Implemented vla-scripts/finetune_capra.py (overlay entrypoint):
  - FinetuneCapraConfig
  - compute_anchor_loss(...)
  - compute_weighted_safer_loss(...)
  - compute_capra_total_loss(...)
  - run_tiny_capra_smoke(...) and run_from_supervision_file(...)
  - CLI with --tiny_smoke for tiny-batch run-through.
- Added test:
  - tests/capra/test_finetune_capra_smoke.py

Tests run:
- pytest tests/capra/test_finetune_capra_smoke.py -q
- Result: 2 passed.
- pytest tests/capra -q
- Result: 13 passed.

Tiny-batch run-through:
- python vla-scripts/finetune_capra.py --tiny_smoke --steps 1
- Log includes both anchor and CAPRA losses:
  - anchor_loss=...
  - capra_loss=...

Upstream hook note:
- No upstream code hook was required in this phase.
- baseline vla-scripts/finetune.py remains unchanged.

Constraints check:
- No discrete logits route.
- No DPO/pairwise ranking.
- No full q-star KL projection.
- No precursor/buffer/merged dataset builder.

Remaining risks/blockers:
- Current finetune_capra.py is a minimal overlay wiring path and tiny-smoke loop, not full distributed production training parity.
- Full RLDS/OpenVLA module wiring can be expanded in later phases if needed, while keeping baseline entry intact.

Next step:
- Phase 5: metrics and evaluation integration (SPIR/EAR + displacement/severe-event summaries).

## Phase 5 - Metrics + Eval
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/metrics.py with v1 metrics:
  - SPIR
  - EAR
  - success_rate
  - displacement_total_mean
  - non_target_displacement_mean
  - severe_event_rate
  - num_effective_steps / num_episodes
- Implemented experiments/robot/capra/run_capra_eval.py:
  - tiny/smoke eval support
  - structured metric dict output and JSON write
- Added script entrypoint:
  - scripts/capra/eval_capra_v1.sh
- Added tests:
  - tests/capra/test_metrics_v1.py

Metric definition alignment:
- SPIR: ratio over effective steps where base is not safest in progress-preserving set.
- EAR: mean positive local regret over effective steps.
- success/external safety metrics aggregated at episode level.
- support-break exact metric is not treated as headline metric in v1.

Tests run:
- pytest tests/capra/test_metrics_v1.py -q
- Result: 3 passed.
- pytest tests/capra -q
- Result: 16 passed.

Eval smoke run:
- python -m experiments.robot.capra.run_capra_eval --supervision_path <synthetic_jsonl> --output_path <tmp_json> --tiny
- Output contains structured metrics dict with SPIR/EAR/success/displacement/severe-event fields.

Constraints check:
- No EditGain implementation.
- No LeadTime implementation.
- No precursor metrics implementation.
- No large benchmark plumbing added.

Remaining risks/blockers:
- Current eval runner is intentionally tiny/smoke-oriented; full benchmark integration is deferred.

Next step:
- CAPRA v1 staged pipeline is now end-to-end closed on scaffold + smoke path.

## Phase 6 - Thin Benchmark Adapters
Status: completed

Completed in this phase:
- Implemented experiments/robot/capra/benchmark_adapters.py:
  - get_libero_utility_eval_command(...) to keep upstream LIBERO utility path explicit.
  - smoke_run_safelibero(...) thin SafeLIBERO suite loader smoke path.
  - smoke_run_custom_split(...) for two custom split names only:
    - support-critical-neighbor
    - chain-reaction
- Updated experiments/robot/capra/run_capra_eval.py:
  - Added benchmark_mode selector:
    - tiny
    - safelibero
    - custom_split
  - Added adapter-mode CLI args without changing baseline upstream eval entry.
- Updated scripts/capra/eval_capra_v1.sh:
  - Added optional adapter-mode arguments while preserving tiny default behavior.
- Added adapter usage/install doc:
  - docs/CAPRA_BENCHMARK_ADAPTERS.md
- Added tests:
  - tests/capra/test_benchmark_adapters.py

Tests run:
- pytest tests/capra -q
- Result: 20 passed.

Smoke runs:
- python -m experiments.robot.capra.run_capra_eval --benchmark_mode custom_split --custom_split support-critical-neighbor --output_path tmp/capra/eval_adapter_custom_split.json
- Output: adapter_ok=1.0 with split metadata.
- python -m experiments.robot.capra.run_capra_eval --tiny --output_path tmp/capra/eval_tiny_phase6.json
- Output: structured tiny metrics dict (SPIR/EAR/success/displacement/severe-event fields).

Constraints check:
- No heavy benchmark framework introduced.
- No SAB/precursor/full dataset builder introduced.
- Upstream LIBERO utility path remains unchanged and callable via adapter helper command builder.

Remaining risks/blockers:
- SafeLIBERO adapter smoke requires external dependency environment consistency under vlsa-aegis/safelibero.
- This phase intentionally provides thin adapter/smoke support, not full benchmark orchestration.

## Phase 7 - Step1/Step2 Cleanup and Decoupling
Status: completed

Completed in this phase:
- Refactored CAPRA Python layout into functional layers:
  - experiments/robot/capra/adapters/
  - experiments/robot/capra/core/
  - experiments/robot/capra/io/
  - experiments/robot/capra/evaluation/
  - experiments/robot/capra/pipelines/
- Refactored CAPRA shell layout:
  - scripts/capra/mine/
  - scripts/capra/train/
  - scripts/capra/eval/
- Updated all in-repo imports and module entrypoints to new paths.
- Added Chinese explanatory comments for high-touch CAPRA files:
  - vla-scripts/finetune_capra.py
  - experiments/robot/capra/core/local_evaluator.py
  - experiments/robot/capra/core/mining.py
  - experiments/robot/capra/adapters/benchmark_adapters.py
- Added shuffle buffer hard-cap notes in vla-scripts/finetune.py comments.
- Reorganized docs directory and created new docs skeleton for Step 3.
- Archived historical CAPRA docs and removed redundant docs/CAPRA02.md.
- Added docs/CHANGELOG.md to track markdown delete/rewrite/archive decisions.

Tests run:
- Not run in this phase (per environment constraint: local code writing only, no runtime execution).

Static checks:
- get_errors reports no editor errors in modified CAPRA core/adapters/pipelines and finetune_capra.py.

Remaining risks/blockers:
- Archived docs still contain old module/script paths by design for history traceability.
- Need Step 3 to produce full beginner-facing Chinese manuals on top of new docs tree.
