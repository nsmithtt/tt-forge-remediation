# Remediation Summary: lexora_lite-causal_lm-pytorch-Lexora_Medium_7B-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[lexora_lite/causal_lm/pytorch-Lexora_Medium_7B-single_device-inference]

## Result
NO_FIX_NEEDED — original MGD topology failure could not be reproduced; test passes on silicon with CI threshold

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-24 08:54:32.019 | critical |          Always | TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology. Inter-mesh mapping failed after 2 attempt(s). Logical meshes being mapped: [0] (1 total). Physical meshes available: [0] (1 total). Failed mesh pair configurations tried: 1 out of 1 possible combinations. Inter-mesh validation mode: STRICT. Solver error: Mapping validation failed: 1 target node(s) are not mapped to any global node: 0. Failed mesh pairs from previous attempts: [(logical=0, physical=0)].. Either relax pinnings or modify the MGD. If this is unexpected, run ./build/test/tt_metal/tt_fabric/test_system_health to check connectivity. (assert.hpp:104)

## Root cause
Could not reproduce. The MGD topology mapping error ("1 target node(s) are not mapped to any global node: 0") did not occur on the test machine. On the original CI machine (ip-172-31-30-236), several ethernet cores were reporting `TT_FATAL: Chip 0 logical eth core connects to a remote mmio device` — on the local machine the same warnings are caught and the eth cores are skipped, allowing topology mapping to succeed. The original failure was likely a transient hardware state issue on that CI instance. The test ran to completion locally and produced PCC=0.986, which passes the CI threshold of 0.95.

## Fix
No fix required. Test passes on silicon with PCC=0.986 > required 0.95 (CI threshold TTXLA_REQUIRED_PCC=0.95). The model (DeepMount00/Lexora-Medium-7B, 7B parameter LLaMA-style CausalLM) compiled and executed without the MGD topology error.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    69.91s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348 |
