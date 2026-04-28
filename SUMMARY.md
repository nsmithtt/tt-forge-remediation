# Remediation Summary: heretic_3b_i1_gguf-causal_lm-pytorch-heretic_3B_I1_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[heretic_3b_i1_gguf/causal_lm/pytorch-heretic_3B_I1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The originally reported failure `RuntimeError: TT_THROW @ silicon_sysmem_manager.cpp:326` was superseded by this loader error when the test was run with the wrong submodule commit.)

## Root cause
The parent repo's tt-xla submodule pointer had `third_party/tt_forge_models` pinned to commit `0f7b734348`, which predates two loader-layer fixes:

1. **poe_8b_gguf loader** (`45ee5f05d7`): `_patched_load_gguf_checkpoint` had the narrow
   signature `(gguf_path, return_tensors=False)`. Because this loader patches
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module-import time
   (during pytest's test-collection phase, which imports all model loaders), the patched
   symbol is what `transformers/modeling_utils.py` picks up via its local `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`. Transformers 5.2 added a
   `model_to_load` kwarg to that call site, so the narrow-signature patch raised
   `TypeError` for every GGUF model loaded after poe_8b_gguf's loader was imported —
   including heretic.

2. **heretic_3b_i1_gguf loader** (`f353b64d67`): added `_ensure_gguf_metadata()` to
   invalidate the importlib cache and patch `gguf.__version__` when the metadata is
   stale (happens when gguf is installed/uninstalled across test runs by
   RequirementsManager).

3. **gguf dependency** (`41a3d29acc`): added `gguf` to the model's `requirements.txt`.

All three fixes were already committed to the
`worktree-aus-wh-07-tt-xla-dev+nsmith+hf-bringup-start500-4` branch of tt_forge_models.
The remedy was advancing the submodule to that branch tip (`45ee5f05d7`).

## Fix
Checked out `worktree-aus-wh-07-tt-xla-dev+nsmith+hf-bringup-start500-4` in
`tt-xla/third_party/tt_forge_models`, which contains:

- `heretic_3b_i1_gguf/causal_lm/pytorch/loader.py` — added `_ensure_gguf_metadata()`,
  called from `_load_tokenizer`, `load_model`, and `load_config`.
- `poe_8b_gguf/causal_lm/pytorch/loader.py` — changed `_patched_load_gguf_checkpoint`
  signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`.
- `heretic_3b_i1_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`.

No changes to tt-xla, tt-mlir, or tt-metal.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    1115.50s (0:18:35)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models (submodule pointer advanced to 45ee5f05d7)

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 45ee5f05d779d5ad850a5b09a9a681afd1e24b19 |
