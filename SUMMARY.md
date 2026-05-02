# Remediation Summary: mistralai_ministral_3_8b_reasoning_2512_gguf-causal_lm-pytorch-Ministral-3-8B-Reasoning-2512-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistralai_ministral_3_8b_reasoning_2512_gguf/causal_lm/pytorch-Ministral-3-8B-Reasoning-2512-GGUF-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

## Root cause
Two bugs, fixed in sequence:

**Loader layer — mistral3 GGUF architecture not registered (3 fixes):**
1. `transformers` 5.x does not recognise `"mistral3"` as a GGUF architecture. The Ministral-3-8B-Reasoning-2512-GGUF file declares arch `"mistral3"` and `load_gguf_checkpoint` raises `ValueError: GGUF model with architecture mistral3 is not supported yet.`
2. Multiple loaders (26+) patch `load_gguf_checkpoint` at pytest collection time. By the time our loader's `load_model()` runs, another loader's old-style wrapper (without `**kwargs`) has overwritten our module-level patch. The context manager reinstalls our patch (and `transformers.modeling_utils.load_gguf_checkpoint` which imports lazily) right before `from_pretrained()`.
3. The `_orig_load_gguf_checkpoint` captured at import time was itself another loader's wrapper without `**kwargs`. Walking the `__closure__` chain finds the true `modeling_gguf_pytorch_utils.load_gguf_checkpoint`.

**tt-xla layer — sliding-window slice OOB (Tier A):**
`MistralForCausalLM` with `sliding_window=4096` generates `aten.slice.Tensor(kv, 2, -4095, MAX_INT)` in the KV cache update. XLA's strict bounds checking rejects start=-4095 on a seq_len=128 dimension. PyTorch CPU silently clamps such indices to 0. Added `canonicalize_slice_indices` FX pass to clamp out-of-range negative start values before XLA lowering.

## Fix
**Loader fix** (`tt_forge_models/mistralai_ministral_3_8b_reasoning_2512_gguf/causal_lm/pytorch/loader.py`):
- Register `"mistral3"` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`
- Add `GGUF_TO_FAST_CONVERTERS["mistral3"]` alias
- Patch `load_gguf_checkpoint` to remap `model_type: "mistral3"` → `"mistral"`
- Patch `get_gguf_hf_weights_map` to remap `"mistral"` back to `"mistral3"` for gguf-py tensor name lookup
- Use context manager in `load_model()` that walks `__closure__` to find the true `modeling_gguf_pytorch_utils.load_gguf_checkpoint` and reinstalls all patches (including `transformers.modeling_utils`) around `from_pretrained()`
- Guard `apply_chat_template` for non-Jinja2 templates

**Compiler frontend fix** (`tt-xla/python_package/tt_torch/backend/passes.py` and `backend.py`):
- Added `canonicalize_slice_indices` FX pass that walks graph nodes and clamps `aten.slice.Tensor` start args that are more negative than `-dim_size`
- Called from `torch_pass_pipeline` after `bypass_dtype_promotion_and_redundant_cast`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    478.80s (0:07:58)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/mistralai_ministral_3_8b_reasoning_2512_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0ab4eb7000dda3e9de9ba1a7a99f14947c32c9f5 |
| tt-forge-models | 2183227193 |
