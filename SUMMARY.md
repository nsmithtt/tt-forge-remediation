# Remediation Summary: qwen_2_5_vl_gguf-pytorch-7B_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_2_5_vl_gguf/pytorch-7B_Instruct_GGUF-single_device-inference]

## Result
FAIL — INTERNAL: Error code: 13 (OOM) during compilation of language model graph; root cause is ttmlir-cumsum-shape-overflow-masked-scatter (Tier B)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-cumsum-shape-overflow-masked-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Traceback (abridged):
  third_party/tt_forge_models/qwen_2_5_vl_gguf/pytorch/src/model.py:154: in forward
      outputs = self.model(**inputs)
  transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1336: in forward
      image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
  transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1336: in torch_dynamo_resume_in_forward_at_1336
      image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
  python_package/tt_torch/backend/backend.py:215: in _call_experimental_compile
      self.compiled_graph = bridge.extract_compiled_graph(...)
  torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
      torch_xla.sync(reset_scope=False)
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Four loader bugs were fixed (see Fix section). After all loader fixes the test
progresses to TT silicon inference, at which point the language model forward
fails with INTERNAL:13 (OOM) during the Dynamo compilation probe.

The root cause is in the tt-mlir lowering of `masked_scatter`. In
`Qwen2_5_VLModel.forward`, vision token embeddings are merged into the input
embedding sequence via:

    input_embeds = input_embeds.masked_scatter(
        mask.unsqueeze(-1).expand_as(input_embeds), image_embeds
    )

where `mask` is a [1, ~2300] boolean tensor and `input_embeds` is
[1, ~2300, 3584]. The tt-mlir lowering of `masked_scatter` uses a CumSumOp
over the boolean mask. For this input shape the CumSumOp is inflated to
approximately 40–50 GiB of intermediate data, exhausting device DRAM and
returning INTERNAL:13 during `torch_xla.sync()` in
`partition_fx_graph_for_cpu_fallback`. This is the same bug seen in
KORMo-VL and InternVL3.5-14B (fingerprint: ttmlir-cumsum-shape-overflow-masked-scatter).

## Fix
Loader-layer fixes committed to tt_forge_models at
`07af78840d6a8952e52f0424813b79037efd8e24` on branch
`remediation/qwen_2_5_vl_gguf-pytorch-7B_Instruct_GGUF-single_device-inference-v2`:

1. **`qwen_2_5_vl_gguf/pytorch/loader.py`** (`4795b00161`):
   - Added `patched_get_gguf_hf_weights_map` inside `_patch_qwen2vl_gguf()`:
     `get_gguf_hf_weights_map` reads `hf_model.config.num_hidden_layers` directly,
     but `Qwen2_5_VLConfig` nests it in `text_config.num_hidden_layers`. The patch
     intercepts `qwen2_5_vl` model type and supplies `num_layers` from
     `config.text_config`.
   - Changed `config.use_cache = False` → `config.text_config.use_cache = False`
     and removed `"use_cache": False` from `model_kwargs` to fix
     `TypeError: __init__() got an unexpected keyword argument 'use_cache'`.
   - Added `use_fast=False` to `AutoProcessor.from_pretrained` to suppress
     transformers 5.x `use_fast=True` default for Qwen2.5-VL.

2. **`qwen_2_5_vl_gguf/pytorch/src/model.py`** (`07af78840d`):
   - Added `_patch_qwen2_5_vl_vision_forward()`: replaces
     `Qwen2_5_VisionTransformerPretrainedModel.forward` with a version that
     computes `cu_seqlens` from `grid_thw.tolist()` Python values instead of
     from XLA tensors. TT silicon computes integer ops in bfloat16, rounding
     38×58=2204 → 2208, causing `RuntimeError: split_with_sizes expects
     split_sizes to sum exactly to 2204, but got split_sizes=[2204]`.
   - Added `_patch_qwen2_5_vl_get_image_features()`: replaces
     `Qwen2_5_VLModel.get_image_features` with a version that computes
     `split_sizes` from `image_grid_thw.tolist()` in Python to avoid the same
     bfloat16 rounding (2208 // 4 = 552 instead of 2204 // 4 = 551).

The terminal INTERNAL:13 bug lives in tt-mlir and is not fixed here.

Proposed fix for the Tier B bug: fix the `masked_scatter` lowering in tt-mlir
to avoid the CumSumOp shape explosion, e.g., by using a scatter-based approach
that does not require materializing a full-shape cumsum intermediate.

## Tier B justification
**cross-cutting**: The `masked_scatter` CumSumOp inflation bug affects all
models that use `masked_scatter(bool_mask, features)` with large bool masks and
fixing it requires changes to the tt-mlir StableHLO→TTIR lowering
infrastructure that handles `masked_scatter` / `scatter` ops. Three models are
already confirmed (KORMo-VL, InternVL3.5-14B, Qwen2.5-VL-7B), suggesting the
fix would touch shared lowering infrastructure impacting many models.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    958.98s (0:15:58)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/qwen_2_5_vl_gguf/pytorch/loader.py`
- `tt_forge_models/qwen_2_5_vl_gguf/pytorch/src/model.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a3e18795d978979a95cdf31b0b9137a38a65b94b |
| tt-forge-models | 07af78840d6a8952e52f0424813b79037efd8e24 |
