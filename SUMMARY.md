# Remediation Summary: dpt_swinv2_large-pytorch-tiny_256-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dpt_swinv2_large/pytorch-tiny_256-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
xla-perf-loop-no-sync-lazy-graph-accumulation

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
`_test_e2e_perf` in `tests/infra/testers/single_chip/model/model_tester.py` discards
warmup results with `_ = self._run_on_tt_device(...)` without flushing the pending XLA
lazy computation graph.  In the XLA lazy-evaluation model, each call to the compiled
model enqueues operations; without a synchronization point those operations accumulate
across iterations.  By the second warmup run the accumulated lazy graph grows large
enough that executing it triggers a full recompilation of the entire accumulated
sequence (~40 s per extra run × 5 extra runs ≈ 210 s of extra time), pushing the test
well past the CI timeout.  The same missing sync also makes the e2e timing numbers
incorrect (they measure lazy-graph construction, not actual device execution).

Confirmed by experiment: with `torch_xla.sync(wait=True)` after each run, warmup 1
takes 49 s (compile + execute) and warmup 2–3 take ~1 s each (cached).  Without sync,
warmup 2 runs for 200+ seconds before being killed.

## Fix
Added a `_sync_tt_device()` hook to `ModelTester` (no-op base implementation) that
is called after every warmup and perf run inside `_test_e2e_perf`.
`TorchModelTester` overrides the hook to call `torch_xla.sync(wait=True)`.

Files changed in tt-xla on branch
`remediation/dpt_swinv2_large-pytorch-tiny_256-single_device-inference`:
- `tests/infra/testers/single_chip/model/model_tester.py`
- `tests/infra/testers/single_chip/model/torch_model_tester.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    116.53s (0:01:56)
- Tier A attempts: 1

## Files changed
- tt-xla: tests/infra/testers/single_chip/model/model_tester.py
- tt-xla: tests/infra/testers/single_chip/model/torch_model_tester.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e900abf553f4f422b0483d8b1514dd9d6be04613 |
| tt-forge-models | 9dd55c9e447709875d5813971ad2f0dfaf4565ca |
