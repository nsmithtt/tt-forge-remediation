# Remediation Summary: brooooooklyn_qwen3_5_35b_a3b_unsloth_mlx-causal_lm-pytorch-35B_A3B_unsloth_mlx-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[brooooooklyn_qwen3_5_35b_a3b_unsloth_mlx/causal_lm/pytorch-35B_A3B_unsloth_mlx-single_device-inference]

## Result
FAIL — transient tt-metal eth-core initialization error; model repo is now 404 on HuggingFace, blocking reproduction

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
tt-metal-eth-core-remote-mmio-init-transient

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 21:07:11.805 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device (assert.hpp:104)

## Root cause
The assertion fires in `tt_cluster.cpp:1307-1312` inside `Cluster::get_connected_ethernet_core()`. The function checks that an active Ethernet link on chip 0 connects to a chip within the local cluster (via `ethernet_connections_within_cluster`). On a Blackhole p150b host, eth core (0,8) is active but connects to a remote MMIO device (a chip accessible via MMIO from outside the tt-metal cluster, e.g., on an expansion chassis). The caller assumes all active eth links are intra-cluster, which is wrong when remote MMIO devices are physically connected.

This error is transient: `failure_patterns.yaml` in the CI framework explicitly excludes "connects to a remote mmio device" from the `tt_fatal` failure category. Inspection of `results_main.yaml` shows 80 tests that record this error in `failure_detail` yet ultimately have `status: SILICON_PASS` — they succeeded on retry after device reset. The brooooooklyn test appears to have run on `bh-lb-13-tt-xla-dev-2` at a moment when the device did not recover from a prior crash.

The model repository `Brooooooklyn/Qwen3.5-35B-A3B-unsloth-mlx` now returns HTTP 404 (deleted or made private after the April 23 CI run). Reproduction is blocked.

## Fix
No fix attempted. The Tier B root cause is a tt-metal bug in `Cluster::get_connected_ethernet_core()` (or its callers during fabric initialization) that assumes every active Ethernet link is intra-cluster. The fix would require auditing all callers that enumerate active eth links and guarding against remote-MMIO connections before calling `get_connected_ethernet_core()`. The exact initialization path that triggers the call during model execution is unknown without a reproducible test case.

## Tier B justification
**internal-error-unknown-mechanism** — the precise initialization path (in tt-xla's PJRT device setup, or tt-mlir's compilation, or tt-metal's program dispatch) that calls `get_connected_ethernet_core()` on a remote-MMIO eth core is not yet isolated. The error is transient, depends on hardware state, and cannot be reproduced with the current model (repo is 404). Diagnosis must come before any fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    not-run (model repo 404; cannot reproduce)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
