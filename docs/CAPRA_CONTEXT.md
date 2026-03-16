# CAPRA Context (Persistent Project Notes)

## Project Identity
- Target: intrinsic safety for manipulation-only VLA.
- Core object: local avoidable-risk regret.
- Not in scope as primary method: test-time shield, memory module, mechanistic steering, generic DPO alignment.

## CAPRA-v1 Required Pipeline
1. Roll out base OpenVLA-OFT in simulator.
2. Build finite local proposal set around base action chunk.
3. Run short counterfactual rollout per candidate from same snapshot.
4. Compute progress features and apply progress-preserving gate.
5. Compute footprint v1.
6. Select safer candidate inside progress-preserving set.
7. Mine safer supervision records.
8. Train CAPRA policy with anchor + weighted safer-target regression.
9. Evaluate SPIR, EAR, success, displacement, severe events.

## Math Objects (v1)
- Candidate set: A_t = {a_base} union local proposals.
- Progress-preserving set: E_t = {a in A_t | G_t(a) >= G_t(a_base) - epsilon_p}.
- Safer candidate: a_star = argmin_{a in E_t} F_t(a).
- Local regret: Delta_t = F_t(a_base) - F_t(a_star).
- Valid supervision condition: E_t non-empty and Delta_t > delta_min.
- Weight: w_t = clip(Delta_t, 0, w_max).
- Loss: L = L_anchor + lambda * w_t * L1(pred_action, a_star).

## Progress and Footprint Boundaries
- Progress v1 is only for obvious non-degradation gate, not exact task equivalence oracle.
- Preferred progress features: progress_before/after/delta, done_after, target_dist_before/after if stable, grasp_held, object_settled, unrecoverable, gripper_open_fraction.
- Footprint v1 includes only displacement_total, severe_event_flags, severe_penalty.
- Default footprint: F_t(a) = displacement_total(a) + severe_penalty(a).

## Action Proposal Constraints
- LIBERO path is continuous action chunk + L1 regression.
- Do not assume discrete logits or policy density sampling.
- Proposal set should stay finite, stable, small, and configurable.

## Integration Rules
- Overlay-only placement:
  - experiments/robot/capra/
  - scripts/capra/
  - tests/capra/
  - vla-scripts/finetune_capra.py (future phase)
- Keep upstream baseline files minimally touched and independently runnable.
- No legacy compatibility shims in this clean upstream repo.

## Signal Access Rules
- obs-first, sim-backed.
- Use env_adapter.get_sim() as the only sim entry.
- Use state_api.read_state_signals() as unified state read layer.
- Business logic must not access deep internal simulator path directly.

## Explicitly Deferred in v1
- test-time shield/safety layer/QP/CBF
- memory main module
- generic DPO mainline
- exact predicate hook, full V_prog
- contact impulse, precursor attribution
- Safety Alternative Buffer, merged dataset builder (heavy)
- real-world/ALOHA mainline

## Workflow Notes from User Context
- Keep CAPRA files grouped and separated from upstream baseline.
- Keep docs under docs/ when relevant to CAPRA planning and context.
- Prefer minimal, reversible changes and avoid unnecessary abstractions.
- If phase prompt and repo reality diverge, document first before extending implementation.
