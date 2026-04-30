# Remediation Summary: gaiasky_qwen_3_5_gguf-causal_lm-pytorch-4B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gaiasky_qwen_3_5_gguf/causal_lm/pytorch-4B_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5 hybrid SSM+attention GGUF has no tensor mapping in transformers; loading as qwen3 fails with missing/mismatched weights

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-hybrid-gguf-no-transformers-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.

Weight report (abridged):
  model.layers.{0...30}.self_attn.q_proj.weight  | MISSING
  model.layers.{0...30}.self_attn.k_proj.weight  | MISSING
  model.layers.{0...30}.self_attn.v_proj.weight  | MISSING
  model.layers.{0...30}.self_attn.o_proj.weight  | MISSING
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.q_proj.weight | MISMATCH ckpt [8192,2560] vs model [2048,2560]
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.k_proj.weight | MISMATCH ckpt [1024,2560] vs model [512,2560]
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.v_proj.weight | MISMATCH ckpt [1024,2560] vs model [512,2560]
  model.layers.{3,7,11,15,19,23,27,31}.self_attn.o_proj.weight | MISMATCH ckpt [2560,4096] vs model [2560,2048]

## Root cause
Three cascading loader bugs were fixed; the terminal failure is a Tier B architecture gap.

**Fixed loader bugs:**

1. `ValueError: GGUF model with architecture qwen35 is not supported yet.`
   The GGUF file declares `general.architecture='qwen35'` which was not in
   `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`.
   Fixed by patching all four transformers module attributes that hold
   `load_gguf_checkpoint` and registering `qwen35` entries that alias the
   existing `qwen3` entries.

2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
   Many other qwen3.5 GGUF loaders in this repo define a narrow-signature
   wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`.
   During pytest collection ALL loaders are imported; the last narrow-sig
   loader to run overwrites the module attribute with its own function. Our
   wrapper's `_orig_load_gguf_checkpoint` was also bound to another loader's
   narrow-sig function (not the real transformers implementation) because
   module attributes are captured at import time. When transformers 5.2.0
   calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, both the
   clobbering problem and the stale binding cause TypeError.
   Fixed by: (a) re-applying patches just before every transformers API call,
   and (b) walking the `__globals__` and `__closure__` chain at call time to
   find the genuine transformers `load_gguf_checkpoint` (identified by
   `__module__ == "transformers.modeling_gguf_pytorch_utils"` and
   `__name__ == "load_gguf_checkpoint"`), storing it in a mutable holder.

**Terminal Tier B failure:**
   Qwen3.5 4B is a hybrid SSM+attention model: 24 GatedDeltaNet (GLA) layers
   plus 8 full-attention layers (every 4th, indices 3,7,11,…,31). The GGUF
   declares `qwen35` architecture. Transformers has `Qwen3_5ForCausalLM` and
   a `qwen3_5` config type, but has NO GGUF support for this architecture:
   `qwen3_5` is absent from `GGUF_CONFIG_MAPPING`, `GGUF_TO_TRANSFORMERS_MAPPING`,
   and `GGUF_TO_FAST_CONVERTERS`. Our patch maps `qwen35` → `qwen3` (Qwen3ForCausalLM),
   which expects all 32 layers to be full self-attention. The result is that
   all GLA layers have MISSING attention weights and the full-attention layers
   have size mismatches (different head config). The correct mapping would
   require implementing complete GGUF tensor-name translations for all
   GatedDeltaNet layer types.

## Fix
**Loader fixes applied** in `tt-forge-models` on branch
`remediation/gaiasky_qwen_3_5_gguf-causal_lm-pytorch-4B_GGUF-single_device-inference`:

- `gaiasky_qwen_3_5_gguf/causal_lm/pytorch/loader.py`: Added `_patch_qwen35_support()`
  to register `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES` and
  `GGUF_TO_TRANSFORMERS_MAPPING` as an alias for `qwen3`.
- `gaiasky_qwen_3_5_gguf/causal_lm/pytorch/requirements.txt`: Added `gguf>=0.10.0`.
- `gaiasky_qwen_3_5_gguf/causal_lm/pytorch/loader.py`: Added `_apply_gguf_patches()`
  helper called at the start of `load_model()`, `_load_tokenizer()`, and
  `load_config()` to re-apply patches just before each transformers API call.
- `gaiasky_qwen_3_5_gguf/causal_lm/pytorch/loader.py`: Added `_find_real_load_gguf()`
  that walks `__globals__` (for module-level captures) and `__closure__` (for
  closure captures like `glm_4_32b_0414_gguf`) to find the genuine transformers
  `load_gguf_checkpoint` regardless of import order.

**Proposed fix for the terminal bug** (not implemented — Tier B):
Add `qwen3_5` support to transformers' GGUF loader:
- `transformers/modeling_gguf_pytorch_utils.py`: Add `qwen3_5` entries to
  `GGUF_CONFIG_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`, and
  `GGUF_TO_TRANSFORMERS_MAPPING` covering all GatedDeltaNet tensor names.
- `transformers/integrations/ggml.py`: Add `qwen3_5` to `GGUF_TO_FAST_CONVERTERS`.
  This is a substantial new feature in transformers upstream.

## Tier B justification
new-infrastructure — Adding GGUF support for `qwen3_5` requires implementing
complete tensor-name mappings for a new hybrid SSM+attention architecture
(GatedDeltaNet layers) that has never been supported in the GGUF loader.
This spans multiple dicts in transformers' GGUF infrastructure and requires
understanding the undocumented GGUF tensor naming convention for GLA/SSM weights.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    412.78s (0:06:52) on last run before terminal failure
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/gaiasky_qwen_3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/gaiasky_qwen_3_5_gguf/causal_lm/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 91bad1e6917598f79ac41247f8b0a9ba730e2fda |
| tt-forge-models | 1f70c40cc6948aa2f8e877e9c0d86dca0de8937c |
