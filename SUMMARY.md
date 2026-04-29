# Remediation Summary: caduceus/masked_lm/pytorch-kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[caduceus/masked_lm/pytorch-kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
caduceus-tie-weights-wrong-cache-path-and-missing-lm-head-tying

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: CaduceusForMaskedLM.tie_weights() got an unexpected keyword argument 'recompute_mapping'
```
(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a harmless SWIG warning that appears in the pytest summary line; the actual test failure is the TypeError above.)

## Root cause
Two bugs in `_patch_caduceus_tie_weights` in the Caduceus loader:

**Bug 1 — wrong cache path patched.**
The function used `try_to_load_from_cache()` to locate the remote model file, which returns a path inside `~/.cache/huggingface/hub/.../blobs/` (via a snapshot symlink). However, `transformers` with `trust_remote_code=True` imports modules from a *separate* copy in `~/.cache/huggingface/modules/transformers_modules/`. The two paths have different inodes. Patching the hub blob has no effect on the module that Python actually imports, so the original `tie_weights(self):` signature (without `**kwargs`) remained in the imported code, causing the `TypeError` from transformers 5.x which passes `recompute_mapping=False`.

**Bug 2 — lm_head not tied to word embeddings (pcc=nan).**
After fixing Bug 1, the model loaded but produced `pcc=nan`. The `CaduceusForMaskedLM.tie_weights` non-RCPS path calls `super().tie_weights(**kwargs)`. In transformers 5.2.0, `PreTrainedModel.tie_weights` checks `getattr(self.config, "tie_word_embeddings", False)` — defaulting to `False` when the attribute is absent. `CaduceusConfig` does not set `tie_word_embeddings`, so the lm_head→word_embeddings weight tying was silently skipped. The lm_head had randomly-initialised weights, producing constant (or NaN) logits and a denominator of zero in the PCC formula.

## Fix
In `tt_forge_models/caduceus/masked_lm/pytorch/loader.py`, the `_patch_caduceus_tie_weights` function was rewritten:

1. **Target the correct cache location.** Replaced `try_to_load_from_cache()` with a glob over `~/.cache/huggingface/modules/transformers_modules/<sanitized_org>/<sanitized_repo>/*/modeling_caduceus.py` (where `-` → `_hyphen_` per transformers' sanitization rules). After writing the patched file, stale `.pyc` files are removed and the module key is evicted from `sys.modules` to force re-import from the patched source.

2. **Explicitly tie lm_head weights.** In the patched non-RCPS `else` branch, added `self.lm_head.weight = self.get_input_embeddings().weight` before the `super().tie_weights(**kwargs)` call so the tying is unconditional regardless of `config.tie_word_embeddings`.

File changed: `tt_forge_models/caduceus/masked_lm/pytorch/loader.py`  
Remediation branch: `remediation/caduceus-masked_lm-pytorch-kuleshov-group-caduceus-ph_seqlen-131k_d_model-256_n_layer-16-single_device-inference` in `tt_forge_models`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    324.09s (0:05:24)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/caduceus/masked_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 784a810566c632024c9ce48cc925f32d947d8839 |
| tt-forge-models | d619f0a2823599fe132b312edb3f58cd9961a1b0 |
