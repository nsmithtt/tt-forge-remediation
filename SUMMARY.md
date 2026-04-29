# Remediation Summary: deepseek_r1_distill_qwen_14b_uncensored_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Qwen_14B_Uncensored_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_14b_uncensored_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Qwen_14B_Uncensored_GGUF-single_device-inference]

## Result
NO_FIX_NEEDED — test passes cleanly on the configured branch; original ARC timeout could not be reproduced

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
what():  Timeout waiting for ARC msg request queue.

## Root cause
The original failure was recorded on machine ip-172-31-30-236 with branch ip-172-31-30-236-tt-xla-dev/ubuntu/hf-bringup-7. On our P150b (Blackhole) machine running the same branch, the test loads the 8.4 GB Q4_K_M GGUF file, compiles cleanly via tt-mlir, and produces matching inference results. No hang or ARC timeout was observed. The ARC timeout on the original machine was likely a transient hardware event or stale device state from a preceding test in the same session.

An earlier reproduction attempt (with tt_forge_models pinned to the pre-existing remediation-branch tip 0f7b734348 instead of the requested hf-bringup-7 branch) exposed a separate loader-layer bug: other GGUF loaders in that older commit tree patched `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at collection time with a narrow signature `(gguf_path, return_tensors=False)` that rejected the transformers 5.x `model_to_load` kwarg. That bug was already fixed in hf-bringup-7 (commit 1571d5f1f6 "Forward all args in _patched_load_gguf_checkpoint wrappers"), so it is not the cause of the ARC timeout and no further action is needed.

## Fix
No fix required. Test was run verbatim on the specified branch and passed.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    553.97s (0:09:13)
- Tier A attempts: N/A

## Files changed
none

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 1571d5f1f63ab8de9e451fa7ff35f99a493554fd |
