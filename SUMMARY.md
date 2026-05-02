# Remediation Summary: kldzj_gpt_oss_120b_heretic_v2_gguf-causal_lm-pytorch-KLDZJ_GPT_OSS_120B_HERETIC_V2_MXFP4_MOE_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kldzj_gpt_oss_120b_heretic_v2_gguf/causal_lm/pytorch-KLDZJ_GPT_OSS_120B_HERETIC_V2_MXFP4_MOE_GGUF-single_device-inference]

## Result
XFAIL — 120B model requires ~240 GB BF16 after MXFP4 dequantization, exceeding single-device DRAM capacity

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
mxfp4-gguf-model-exceeds-single-device-dram

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
The `kldzj/gpt-oss-120b-heretic-v2` model has 120 billion parameters. The GGUF file uses MXFP4
quantization for MoE expert weights (2 shards, ~59 GB total on disk). TT hardware has no native
FP4/MXFP4 execution support, so the model must be dequantized to BF16 at load time.

At BF16: 120B parameters × 2 bytes = 240 GB, which far exceeds the DRAM capacity of any single
TT device (p150b or otherwise). The original CI failure was a test timeout from downloading the
59 GB GGUF files on a slow network connection before even reaching the device.

When the GGUF files are locally cached, the test fails immediately with:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
This secondary error is a cross-cutting loader bug (Tier B): 26 loaders each patch
`load_gguf_checkpoint` globally at import time using a strict 2-argument signature
`(gguf_path, return_tensors=False)` that omits the `model_to_load` kwarg added in transformers
5.x. The last such loader imported during test collection replaces the global reference, causing
any subsequent `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` call to fail.
Fingerprint: `gguf-load-checkpoint-model-to-load-kwarg`. This is Tier B (>25 files, cross-cutting).
Even if fixed, the hardware capacity ceiling applies.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for this test in:
`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

No compiler stack changes were needed or made.

## Verification
- pytest exit: FAIL (TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')
- Hardware:    not-run (failed before device execution)
- Duration:    ~4 minutes (GGUF shard 1 download + TypeError)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4e17eab04df86e143a107054a984e14a637b02b8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
