# Remediation Summary: 4_5test_gguf-causal_lm-pytorch-4_5test_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[4_5test_gguf/causal_lm/pytorch-4.5test_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — GLM-4.5 Air (128-expert MoE, ~120B total params, 68 GB GGUF) exceeds single-device n150 DRAM (12 GB); hardware capacity ceiling.

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm4-5-air-128-expert-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: Test exceeded configured timeout and was killed

Reproduced locally as:
```
ValueError: GGUF model with architecture glm4moe is not supported yet.
```

The `4.5test.Q4_K_M.gguf` file (68 GB, mradermacher/4.5test-GGUF) encodes the
GLM-4.5 Air model, whose GGUF metadata uses the architecture identifier
`glm4moe`. Transformers 5.x has `Glm4MoeForCausalLM` (model_type=`glm4_moe`)
but no GGUF config mapping or weight-tensor mapping for `glm4moe`, so
`AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` raises a `ValueError`
before attempting any device operations.

After fixing the loader to register `glm4moe`, the dequantization step
("Converting and de-quantizing GGUF tensors... 0/803") was observed to project
~76 minutes completion time, consistent with the CI timeout. Even if dequantization
completed, the model cannot run on device: 128 experts × 3 matrices × 4096 ×
1408 × bfloat16 × 47 layers ≈ 240 GB weight data, far exceeding n150 DRAM (12 GB).

## Root cause
Two independent issues:

1. **Loader bug**: `glm4moe` (the GGUF architecture name for GLM-4.5 Air) is not
   registered in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`,
   `GGUF_TO_FAST_CONVERTERS` (tokenizer), or the tensor-name mapper
   `get_gguf_hf_weights_map`. The `4_5test_gguf` loader had no patch to bridge
   the `glm4moe` → `glm4_moe` gap.

2. **Hardware capacity ceiling**: GLM-4.5 Air has 128 experts totalling ~120B
   parameters. The Q4_K_M GGUF file is 68 GB; dequantized to bfloat16 it is
   ~240 GB — 20× the n150 single-device DRAM budget (12 GB). Single-device
   inference is architecturally impossible.

## Fix
**Loader fix** (`tt_forge_models`, branch
`remediation/4_5test_gguf-causal_lm-pytorch-4_5test_Q4_K_M_GGUF-single_device-inference`,
commit `8eac0ff9485f04f7f3c856452635d5e82e60ec28`):

Added `_patch_transformers_glm4moe_gguf()` in
`4_5test_gguf/causal_lm/pytorch/loader.py`:
- Appends `"glm4moe"` to `GGUF_SUPPORTED_ARCHITECTURES`.
- Adds full MoE config field mapping for `glm4moe` in
  `GGUF_TO_TRANSFORMERS_MAPPING["config"]` (block_count, embedding_length,
  expert_count, expert_feed_forward_length, etc.).
- Registers `GGUFQwen2Converter` as the tokenizer converter for `glm4moe`.
- Patches `load_gguf_checkpoint` to remap `model_type` from `"glm4moe"` to
  `"glm4_moe"` and compute `partial_rotary_factor` from
  `rope.dimension_count / head_dim` (= 64/128 = 0.5).
- Patches `get_gguf_hf_weights_map` to remap `"glm4_moe"` → `"glm4moe"` for
  the `gguf-py MODEL_ARCH_NAMES` tensor-name lookup.

**Test config** (`tt-xla`, same branch):

Added `KNOWN_FAILURE_XFAIL` entry for
`4_5test_gguf/causal_lm/pytorch-4.5test_Q4_K_M_GGUF-single_device-inference` in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: not-run (hardware capacity prevents reaching device)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/4_5test_gguf/causal_lm/pytorch/loader.py` — add `_patch_transformers_glm4moe_gguf()`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | de954f10bc7d233e21864f0a7e0768946818ca2c |
| tt-forge-models | 8eac0ff9485f04f7f3c856452635d5e82e60ec28 |
