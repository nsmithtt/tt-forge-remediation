# Remediation Summary: ministral_3_3b_reasoning_2512_gguf-causal_lm-pytorch-Ministral-3-3B-Reasoning-2512-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ministral_3_3b_reasoning_2512_gguf/causal_lm/pytorch-Ministral-3-3B-Reasoning-2512-GGUF-single_device-inference]

## Result
SILICON_PASS

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
Reported CI failure: `raise InvalidVersion(f"Invalid version: {version!r}")` — not reproduced locally.

Actual failures encountered:
1. `ValueError: GGUF model with architecture mistral3 is not supported yet.`
2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` (session contamination)
3. `RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095) — While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807))` (tt-xla Tier A)

## Root cause

**Bug 1 — loader (mistral3 arch not registered):**
gguf-py 0.18.0 defines `MODEL_ARCH_NAMES[MISTRAL3] = "mistral3"`, but transformers 5.x does not have `mistral3` in `GGUF_CONFIG_MAPPING` or `GGUF_SUPPORTED_ARCHITECTURES`. Additionally, `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` maps `mistral3 → Mistral3ForConditionalGeneration` (a VLM), not `MistralForCausalLM`. Fix: register `mistral3` as alias for `mistral` in all three mapping tables; remap `model_type=mistral3→mistral` in `load_gguf_checkpoint`; remap back `mistral→mistral3` in `get_gguf_hf_weights_map` for gguf-py tensor name lookup.

**Bug 2 — loader (session contamination via narrow-sig patchers):**
27 GGUF loaders (bartowski, daniloreddy, dmind, gpt_oss, 16 mradermacher, qwen_3_5_imatrix, tvall43 ×2, unified_reward_flex) patch `_gguf_utils.load_gguf_checkpoint` at module import time. Loaders alphabetically after `ministral_*` replaced the ministral loader's patch with narrow-sig versions that omit `**kwargs`. When `transformers/modeling_utils.py:4016` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the last narrow-sig loader's function was active.

**Bug 3 — tt-xla (XLA slice OOB):**
Mistral uses sliding window attention with `window=4096`. On short test inputs (seq_len=8), `cache_utils.py:214` generates `full_kv[:, :, -4095:, :]`. PyTorch eager silently clamps out-of-range negative slice starts; the XLA lazy backend raises "Value out of range" instead. Fix: an FX pass that clamps negative `start` values in `aten.slice.Tensor` nodes to `-dim_size` when `start < -dim_size`.

## Fix

**Loader fixes** (`tt_forge_models`, remediation branch):
- `ministral_3_3b_reasoning_2512_gguf/causal_lm/pytorch/loader.py`: register `mistral3` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`; patch `load_gguf_checkpoint` to remap `model_type`; patch `get_gguf_hf_weights_map` to re-remap for gguf-py lookup.
- 26 other GGUF loader files: add `**kwargs` to `_patched_load_gguf_checkpoint` signature and forward to `_orig_load_gguf_checkpoint`.

**Compiler fix** (`tt-xla`, remediation branch):
- `python_package/tt_torch/backend/passes.py`: added `clamp_out_of_range_slice_starts(gm)` FX pass.
- `python_package/tt_torch/backend/backend.py`: import and call `clamp_out_of_range_slice_starts` after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 384.04s (0:06:24)
- Tier A attempts: 1

## Files changed
- `ministral_3_3b_reasoning_2512_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- 26 other GGUF loaders with narrow-sig `_patched_load_gguf_checkpoint` (tt_forge_models)
- `python_package/tt_torch/backend/passes.py` (tt-xla)
- `python_package/tt_torch/backend/backend.py` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2d8dceb7aa1d1f03bd24d422da8d273693b374e6 |
| tt-forge-models | 8e24c4887fd20d91b6dd93de1c65bb4c39a5828b |
