# Remediation Summary: darkhn_quants_3_qwen3_5_9b_animus_v13_0_gguf-causal_lm-pytorch-9B_Animus_V13.0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkhn_quants_3_qwen3_5_9b_animus_v13_0_gguf/causal_lm/pytorch-9B_Animus_V13.0_GGUF-single_device-inference]

## Result
FAIL — Tier B new-infrastructure: qwen35 hybrid SSM+GLA GGUF architecture has no tensor name mapping in transformers

## Stack layer
loader

## Tier
B

## Bug fingerprint
gguf-qwen35-hybrid-no-tensor-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AttributeError: 'NoneType' object has no attribute 'config'

(This was the original failure reported. During debugging, two loader bugs were fixed before reaching the underlying architecture mismatch. The final failure after those fixes is:)

E   RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!

(Shape mismatches: layers 3,7,11,15,19,23,27,31 q_proj weight (8192, 4096) vs expected (2048, 4096); 24 GLA layers have no matching weights at all.)

## Root cause
**Original failure** (`AttributeError: 'NoneType' object has no attribute 'config'`): Two chained loader bugs:

1. **`_patched_load_gguf_checkpoint()` gets `model_to_load` kwarg** — transformers 5.2.0 added a `model_to_load` parameter to `load_gguf_checkpoint()`. When `from_pretrained` calls this with `return_tensors=True`, it passes `model_to_load=dummy_model`. Our module-level `_apply_patches()` was overwritten by later-imported loaders. At `load_model()` time, `_gguf_utils.load_gguf_checkpoint` still pointed to an old-pattern patcher (no `*args, **kwargs`), which rejected the kwarg. Fixed by re-applying patches at `load_model()` time.

2. **Closure-based chain traversal** — `_find_real_load_gguf_checkpoint()` traversed the patcher chain looking for the real transformers function, but the `onion008` loader captures the previous function as a closure variable (`orig_load = gguf_utils.load_gguf_checkpoint`) rather than a module global. The traversal stopped at `onion008.patched_load_gguf_checkpoint` (which lacked `model_to_load`), causing the same TypeError. Fixed by extending traversal to inspect `fn.__closure__` via `fn.__code__.co_freevars`, and using `importlib.util.find_spec` for exact-file matching.

**Remaining failure** (after above fixes): The Qwen3.5 9B Animus model uses a **hybrid SSM+GLA architecture** (`Qwen3_5TextConfig`, `model_type=qwen3_5_text`). Only 1 in 4 layers (positions 3, 7, 11, 15, 19, 23, 27, 31) are standard full-attention (`self_attn`). The other 24 layers are GLA (Gated Linear Attention / `linear_attn`). The GGUF file declares `general.architecture = "qwen35"` and carries tensor names specific to this hybrid layout.

Our patch maps `qwen35 → qwen3` (standard full-attention Qwen3). `Qwen3ForCausalLM` expects 32 identical full-attention layers. The GGUF has full-attention weights for only 8 positions, and GLA weights (under `linear_attn.*` tensor names) for the remaining 24. Result: 8 shape mismatches + 24 layers with missing weights.

`transformers` 5.2.0 does have `Qwen3_5ForCausalLM` and `Qwen3_5TextConfig` (model_type `qwen3_5_text`), but there is **no GGUF loading support**: `qwen3_5_text` is absent from `GGUF_SUPPORTED_ARCHITECTURES` and from `GGUF_TO_TRANSFORMERS_MAPPING`. The GGUF tensor names for GLA layers (e.g., `blk.N.linear_attn.*`) have no defined mapping to PyTorch parameter names.

## Fix
The two chained loader bugs (chain traversal stopping at closure-based patchers) are fixed in
`tt-forge-models/darkhn_quants_3_qwen3_5_9b_animus_v13_0_gguf/causal_lm/pytorch/loader.py`:

- **Commit `d8af1b2c90`**: Register `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING` (as alias for `qwen3`) to unblock GGUF file loading.
- **Commit `17cd1759a4`**: Add `_find_real_load_gguf_checkpoint()` that traverses the patcher chain via `fn.__globals__` (module-level captures) and rebuild with `_apply_patches()` called at `load_model()` time.
- **Commit `737ba3c96b`**: Extend traversal to also inspect `fn.__closure__` / `fn.__code__.co_freevars` for closure-captured `orig_load` variables, and use `importlib.util.find_spec` for exact-file matching to reliably identify the real transformers function.

The remaining bug requires new infrastructure in transformers or a standalone GGUF weight mapping:

**Proposed fix (requires new infrastructure):** Add `qwen3_5_text` / `qwen35` to `transformers.modeling_gguf_pytorch_utils`:
- `GGUF_SUPPORTED_ARCHITECTURES`: add `"qwen35"` pointing to `Qwen3_5ForCausalLM`
- `GGUF_TO_TRANSFORMERS_MAPPING["config"]`: map all `qwen35` config fields to `Qwen3_5TextConfig` fields (including `full_attention_interval`, `layer_types`, `linear_attn_*` parameters)
- `GGUF_TO_TRANSFORMERS_MAPPING["tensors"]` or `get_gguf_hf_weights_map()`: map `blk.N.linear_attn.*` GGUF tensor names to the GLA layer parameter names in `Qwen3_5TextModel`
- `GGUF_TO_FAST_CONVERTERS`: register `"qwen35"` with a tokenizer converter

This is effectively reimplementing the full GGUF→transformers bridge for a novel hybrid architecture with undocumented GGUF tensor key names for the GLA layers.

## Tier B justification
**new-infrastructure**: Adding complete GGUF tensor name mapping for a hybrid SSM+GLA architecture requires: (1) discovering and mapping all GLA-layer GGUF tensor key names (undocumented), (2) updating `get_gguf_hf_weights_map()` with the hybrid `Qwen3_5TextModel` parameter layout, and (3) registering config field mappings for `Qwen3_5TextConfig`-specific fields. This is new infrastructure across multiple files and functions in `transformers.modeling_gguf_pytorch_utils` — not a scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: not-run (failed before compilation)
- Duration: 393.53s (6:33) — time spent downloading GGUF and diagnosing
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/darkhn_quants_3_qwen3_5_9b_animus_v13_0_gguf/causal_lm/pytorch/loader.py` (3 commits: d8af1b2c90, 17cd1759a4, 737ba3c96b)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8ad7f1cc8833ce0111ab6c11928ca8fa6ecf7302 |
| tt-forge-models | 737ba3c96bdb432dce3849396a935a426f2f4efd |
