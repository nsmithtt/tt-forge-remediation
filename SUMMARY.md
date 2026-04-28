# Remediation Summary: bge_reranker_v2_minicpm_layerwise-passage_ranking-pytorch-layerwise-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bge_reranker_v2_minicpm_layerwise/passage_ranking/pytorch-layerwise-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-minicpm-remote-code-compat

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
(actual error: KeyError: 'type' in _init_rope when accessing self.config.rope_scaling["type"])

## Root cause
Four transformers 5.x incompatibilities in the custom model code (loaded via
trust_remote_code=True) and in how transformers 5.x initializes models:

1. **is_torch_fx_available removed**: the cached remote code
   (`modeling_minicpm_reranker.py`) imports `is_torch_fx_available` from
   `transformers.utils.import_utils` at module level; transformers 5.x
   removed this function (torch.fx is now unconditionally available).
   This caused `ImportError` before model init even started.

2. **rope_scaling format change**: transformers 5.x auto-populates
   `rope_scaling=null` in config.json to `{'rope_type': 'default',
   'rope_theta': 10000.0}`. The custom `_init_rope` method checks
   `if self.config.rope_scaling is None` then falls through to
   `self.config.rope_scaling["type"]`, causing `KeyError: 'type'`.

3. **_tied_weights_keys format change**: transformers 5.x expects
   `_tied_weights_keys` to be a dict (`{tied_key: source_key}`), but the
   custom class defines it as a list `["lm_head.weight"]` (4.x format).
   `post_init` → `get_expanded_tied_weights_keys` crashes with
   `AttributeError: 'list' object has no attribute 'keys'`. With
   `head_type=simple` the lm_head is an `nn.ModuleList` of `LayerWiseHead`
   objects (output dim 1, not vocab size) — there is no vocab embedding
   tying, so the class attribute is stale for this config.

4. **Meta-device non-persistent buffers uninitialized**: transformers>=5.0
   uses meta device during `from_pretrained`. Non-persistent buffers
   (`inv_freq`, `cos_cached`, `sin_cached`) are not materialized, leaving
   them with garbage values (`5.35e-26, 0, 0, ...` instead of proper RoPE
   frequencies). This caused both CPU and TT outputs to be NaN, resulting
   in `pcc=nan`.

## Fix
All four fixes applied in the loader (`bge_reranker_v2_minicpm_layerwise/passage_ranking/pytorch/loader.py`) in tt_forge_models:

1. Monkey-patch `transformers.utils.import_utils.is_torch_fx_available = lambda: True`
   before calling `from_pretrained` (so the dynamic module import succeeds).

2. Load config explicitly with `AutoConfig.from_pretrained`, then translate
   `rope_scaling`: reset to `None` if `rope_type == 'default'`, or copy
   `rope_type → type` for other scaling types.

3. Set `config.tie_word_embeddings = False` to prevent transformers from
   attempting to process the stale `_tied_weights_keys` list.

4. After `from_pretrained`, iterate model modules and reinitialize any
   rotary embedding with `inv_freq` + `_set_cos_sin_cache` attributes,
   recomputing `inv_freq` from `base` and `dim`.

Branch: `remediation/bge_reranker_v2_minicpm_layerwise-passage_ranking-pytorch-layerwise-single_device-inference` in tt_forge_models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    130.42s
- Tier A attempts: N/A

## Files changed
- `bge_reranker_v2_minicpm_layerwise/passage_ranking/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0b9efa04b70e76beb24887df44ab14f726ef34b5 |
| tt-forge-models | 2a8d1f2daa01ae4bf72cca1da5430c383e989301 |
