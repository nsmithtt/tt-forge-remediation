# Remediation Summary: aidc_llm_laos_12b_i1_gguf-causal_lm-pytorch-12B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aidc_llm_laos_12b_i1_gguf/causal_lm/pytorch-12B_i1_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: GGUF patchers missing **kwargs (26 loaders) and aten.slice.Tensor OOB start index

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-33, 32], but got -1023)

When run in isolation the test first hits:
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Another GGUF loader imported during pytest collection had already installed a broken patcher with a fixed signature.)

## Root cause
Two independent bugs:

1. **Loader layer (26 GGUF loaders):** pytest imports all loader modules at collection time. Loaders for Qwen3.5 and GPT-OSS Swallow GGUF models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module level with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` — missing `**kwargs`. Transformers 5.x added `model_to_load=None` to this function's signature and now passes it as a kwarg. The broken patcher raises TypeError for any GGUF model test whose loader runs after the patcher is installed.

2. **tt-xla compiler frontend:** The AIDC LLM Laos 12B is a Gemma3-derivative with sliding window attention (sliding_window=1024). `SlidingWindowCache.update()` performs `full_value_states[:, :, -sliding_window+1:, :]` = `full_value_states[:, :, -1023:, :]`. When seq_len=33 (< 1024), the start index -1023 is below `-dim_size=-33`, which PyTorch clamps to 0 but XLA/TT raises `RuntimeError: Value out of range (expected to be in range of [-33, 32], but got -1023)`.

## Fix
1. **tt_forge_models** (`remediation/aidc_llm_laos_12b_i1_gguf-causal_lm-pytorch-12B_i1_GGUF-single_device-inference`): Cherry-picked commit `57caeafc70` which fixes all 26 broken GGUF patchers to use `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs)` and forward `**kwargs` to the original function.

2. **tt-xla** (`remediation/aidc_llm_laos_12b_i1_gguf-causal_lm-pytorch-12B_i1_GGUF-single_device-inference`):
   - `python_package/tt_torch/backend/passes.py`: Added `clamp_out_of_range_slice_starts(gm)` — iterates `aten.slice.Tensor` nodes, checks start against `-dim_size` from `node.args[0].meta["val"].shape`, clamps start to `max(-dim_size, start)`, then calls `gm.recompile()`.
   - `python_package/tt_torch/backend/backend.py`: Added import and call of `clamp_out_of_range_slice_starts` after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `aidc_llm_laos_12b_i1_gguf/causal_lm/pytorch-12B_i1_GGUF-single_device-inference` as `EXPECTED_PASSING`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    722.64s (0:12:02)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer updated)
- 26 × `tt_forge_models/<loader>/causal_lm/pytorch/loader.py` (kwargs fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5e3815729212fd96cf21aeb6d40f3a6bc8dfd054 |
| tt-forge-models | 346c64aff85cc37894c06be213752d99287940fb |
