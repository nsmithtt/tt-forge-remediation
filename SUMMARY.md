# Remediation Summary: audiogemma_3n_finetune_gguf-causal_lm-pytorch-FINETUNE_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[audiogemma_3n_finetune_gguf/causal_lm/pytorch-FINETUNE_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-gemma3n-architecture-missing, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

After adding requirements.txt: `ValueError: GGUF model with architecture gemma3n is not supported yet.`

After loader fix: `RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)` in `aten.slice.Tensor` at `torch_overrides.py:34`

## Root cause
Two independent bugs:

1. **Loader (tt_forge_models)**: `audiogemma_3n_finetune_gguf` had no `requirements.txt`, causing an `ImportError` when the `gguf` package was absent. Additionally, transformers 5.2.0 includes `Gemma3nForCausalLM` but has no GGUF loading support for the `"gemma3n"` architecture name. The GGUF file reports `model_type="gemma3n"` (the multimodal class), which must be remapped to `"gemma3n_text"` (text-only `Gemma3nForCausalLM`). GGUF config fields for rope (two-frequency RoPE for sliding/full attention) and `layer_types` (derived from `attention.sliding_window_pattern`) also require post-processing not present in stock transformers.

2. **tt-xla (Tier A)**: XLA's `aten.slice.Tensor` validates start strictly (raises `ValueError` when `start < -dim_size`), whereas PyTorch CPU silently clamps. Gemma3n's sliding-window cache update computes `full_value_states[:, :, -sliding_window+1:, :]`; with `seq_len=23 < sliding_window=512`, start becomes `-511`, which falls outside the valid range `[-23, 22]`.

## Fix
**Loader fix** — `tt_forge_models` branch `remediation/audiogemma_3n_finetune_gguf-causal_lm-pytorch-FINETUNE_GGUF-single_device-inference` (commit `8181b039b7`):
- `audiogemma_3n_finetune_gguf/causal_lm/pytorch/requirements.txt`: add `gguf>=0.10.0`
- `audiogemma_3n_finetune_gguf/causal_lm/pytorch/loader.py`: add `_patch_transformers_gemma3n_gguf()` called at import time; registers `"gemma3n"` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, `TENSOR_PROCESSORS`, and `GGUF_TO_FAST_CONVERTERS`; wraps `load_gguf_checkpoint` to remap `model_type` and reconstruct `rope_parameters`/`layer_types`; patches `get_gguf_hf_weights_map` to reverse-remap `"gemma3n_text"→"gemma3n"` for gguf-py lookup; adds `ignore_mismatched_sizes=True` to `from_pretrained` (audio encoder weights absent in text-only GGUF).

**tt-xla Tier A fix** — `tt-xla` branch `remediation/audiogemma_3n_finetune_gguf-causal_lm-pytorch-FINETUNE_GGUF-single_device-inference` (commit `1dc0d5e32`):
- `python_package/tt_torch/torch_overrides.py`: in `TorchFunctionOverride.__torch_function__`, intercept `torch.ops.aten.slice.Tensor` and clamp `start` to `max(start, -dim_size)` before dispatch when `start < -dim_size`.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    785.48s (0:13:05)
- Tier A attempts: 1

## Files changed
- `audiogemma_3n_finetune_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- `audiogemma_3n_finetune_gguf/causal_lm/pytorch/requirements.txt` (tt_forge_models, new file)
- `python_package/tt_torch/torch_overrides.py` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1dc0d5e320d7ed0e20addfc8d0ed7f0d0d25084e |
| tt-forge-models | 8181b039b71ef458660921ca10fe005b696d9238 |
