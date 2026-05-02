# Remediation Summary: mlx_community_lfm2_vl_1_6b_4bit/image_text_to_text/pytorch-1_6B_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_lfm2_vl_1_6b_4bit/image_text_to_text/pytorch-1_6B_4bit-single_device-inference]

## Result
XFAIL — pjrt-device-to-host-transfer: Lfm2ShortConv.slow_forward calls cache_position[0] > 0 on a TT tensor, triggering INTERNAL Error code 13

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
ValueError: The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized
```

After loader fix (terminal failure):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
from `torch_xla._XLAC._run_cached_graph` during the language model compiled graph execution. Traceback points into `Lfm2ShortConv.slow_forward` called from `Lfm2HybridConvCache.__init__` at modeling_lfm2.py:648.

## Root cause
**Loader bugs (fixed — 4 bugs + 1 vision patch):**

1. `config.quantization_config` has `{group_size: 64, bits: 4}` but no `quant_method`. transformers 5.x raises ValueError. Fix: delete `config.quantization_config` before `from_config()`.

2. `Lfm2VlForConditionalGeneration._tied_weights_keys = ["lm_head.weight"]` is a list. transformers 5.x `get_expanded_tied_weights_keys` calls `.keys()` on it → AttributeError. Fix: patch to None before model instantiation.

3. `lm_head.weight` is tied to `embed_tokens.weight` and not saved in the MLX checkpoint. After `load_state_dict(strict=False)` with `_tied_weights_keys=None`, lm_head stays randomly initialized. Fix: manually re-tie after load.

4. MLX stores Conv1d weights channel-last [out, K, in] while PyTorch expects [out, in, K]. `conv.conv.weight` shape [2048, 3, 1] vs model [2048, 1, 3] causes RuntimeError. Fix: `permute(0, 2, 1)` on 3-D non-quantized weight tensors.

5. `Siglip2VisionEmbeddings.resize_positional_embeddings` calls `F.interpolate(antialias=True)` which requires float32. Upstream only casts on CPU; on TT bfloat16 tensors raises NotImplementedError. Fix: patch method to cast unconditionally to float32 before interpolate.

**Terminal Tier B bug:**

`Lfm2ShortConv.slow_forward` (transformers `modeling_lfm2.py` line 508) evaluates `if past_key_values is not None and cache_position[0] > 0:`. When `cache_position` is a TT tensor, Python's `if` must materialize `cache_position[0] > 0` as a Python bool → device-to-host read → INTERNAL Error code 13.

## Fix
**Loader** (`tt_forge_models/mlx_community_lfm2_vl_1_6b_4bit/image_text_to_text/pytorch/loader.py`): Rewrote `load_model` to manually dequantize the MLX 4-bit checkpoint. Added `_dequantize_mlx4bit` (uint32 nibble unpack with per-group bf16 scale+bias), `_load_and_dequantize_mlx_vl` (shard loading, key remapping, conv weight permutation), and `_patched_resize_positional_embeddings` (unconditional float32 cast for antialias interpolation). Used `use_fast=False` for the processor.

**Evaluator** (`tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`): Added duck-type detection for non-Cache subclasses with `key_cache`/`value_cache` (handles `Lfm2HybridConvCache`); `numel() > 0` filter for empty conv placeholders.

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `mlx_community_lfm2_vl_1_6b_4bit/image_text_to_text/pytorch-1_6B_4bit-single_device-inference` as `KNOWN_FAILURE_XFAIL`.

**Proposed Tier B fix**: In `Lfm2ShortConv.slow_forward`, replace `cache_position[0] > 0` with a Python-side int extracted before compilation (e.g., pass `is_prefill: bool` as a separate argument from the forward or use `is_torchdynamo_compiling()` to skip the cache branch during tracing). The fix lives in the transformers `modeling_lfm2.py` (or remote model code).

## Tier B justification
new-infrastructure — TT PJRT runtime does not implement synchronous device-to-host tensor reads; any Python-level bool evaluation of a TT tensor (`cache_position[0] > 0` inside a Dynamo-compiled subgraph) raises INTERNAL Error code 13. Implementing this requires changes to the TT PJRT buffer layer.

## Verification
- pytest exit: PASS (1 xfailed)
- Hardware: blackhole-p150b
- Duration: 329.85s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mlx_community_lfm2_vl_1_6b_4bit/image_text_to_text/pytorch/loader.py`
- `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aea9457eda17d162251ec5ae2ac488d6bc2bc142 |
| tt-forge-models | ccddbc861cb95ac0d4944f0037170897ccc49167 |
