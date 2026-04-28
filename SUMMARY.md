# Remediation Summary: babylm_git-pytorch-multimodal_baseline-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[babylm_git/pytorch-multimodal_baseline-single_device-inference]

## Result
SILICON_PASS — all loader bugs fixed and Tier A index-dtype normalization pass added; test passes on n150

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-index-tensor-mixed-int32-int64-indices

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise RuntimeError(
ImportError: This modeling file requires the following packages that were not found in your environment: ipdb. Run `pip install ipdb`
```
(Original manifest failure; subsequent failures listed in Root cause.)

## Root cause
Four cascading bugs, three in the loader and one in the tt-xla compiler frontend:

1. **Loader — missing requirement**: `BabyLM-community/babylm-multimodal-baseline-git`'s remote
   `modeling_git.py` has `import ipdb` at module level (a debug artifact). `transformers`'
   `check_imports` raised `ImportError` before the model could load. Fix: add `ipdb` to
   `babylm_git/pytorch/requirements.txt`.

2. **Loader — transformers 5.x breaking change**: The same remote file also does
   `from transformers import ViTFeatureExtractor`, removed in transformers 5.x. Because
   `transformers` uses a `_LazyModule` whose attribute resolution goes through `_objects` (not
   the module's `__dict__`), the shim must be injected into `_LazyModule._objects` rather than
   the module namespace. Fix: inject `ViTFeatureExtractor = ViTImageProcessor` into
   `sys.modules["transformers"]._objects` before `from_pretrained`.

3. **Loader — meta-device init context with nested `from_pretrained`**: `GitForCausalLM.__init__`
   calls `ViTModel.from_pretrained('facebook/dino-vitb16')` while transformers 5.x runs
   `__init__` under `torch.device("meta")`. The nested `from_pretrained` fails because
   `check_and_set_device_map` raises when the meta-device context is active.
   Additionally, `_move_missing_keys_from_meta_to_device` unconditionally replaces all
   `persistent=False` buffers (including `GitEmbeddings.position_ids`) with
   `torch.empty_like` (uninitialized), producing garbage position indices.
   Fix: patch `PreTrainedModel.get_init_context` to strip `torch.device` entries for this
   `from_pretrained` call, then re-initialize `model.git.embeddings.position_ids` from
   `torch.arange(max_position_embeddings)` after loading.

4. **tt-xla compiler — mixed int32/int64 indices in `aten.index.Tensor`**: In `modeling_git.py`
   (transformers' installed copy), `cache_position` is created with `dtype=torch.int` (int32)
   at line 1103. When vmap expands the masking function over `cache_position`, `q_idx` becomes
   int32. `batch_arange = torch.arange(batch_size)` uses default int64. The resulting
   `token_type_ids[batch_idx, safe_q_idx]` call passes [int64, int32] indices to
   `aten.index.Tensor`. XLA's `stablehlo.concatenate` of the two index tensors fails with
   "Cannot concatenate arrays with different element types: S64 vs S32".
   Fix: new `normalize_index_tensor_index_dtypes` FX pass in `tt_torch/backend/passes.py`
   that inserts `aten._to_copy(dtype=int64)` before any `aten.index.Tensor` whose index list
   contains int32/int16/int8 tensors.

## Fix

**tt-forge-models** (`remediation/babylm_git-pytorch-multimodal_baseline-single_device-inference`):
- `babylm_git/pytorch/requirements.txt` — new file, adds `ipdb` (debug artifact in remote model code)
- `babylm_git/pytorch/loader.py` — three loader fixes:
  - Inject `ViTFeatureExtractor` shim into `transformers._LazyModule._objects`
  - Patch `PreTrainedModel.get_init_context` to remove meta-device context around `from_pretrained`
  - Re-initialize `model.git.embeddings.position_ids` after loading

**tt-xla** (`remediation/babylm_git-pytorch-multimodal_baseline-single_device-inference`):
- `python_package/tt_torch/backend/passes.py` — add `normalize_index_tensor_index_dtypes` pass
- `python_package/tt_torch/backend/backend.py` — wire the new pass into the compilation pipeline
- `tests/runner/utils/dynamic_loader.py` — cherry-pick spacy sys.path shadowing fix (from aesthetic_shadow branch, bug `spacy-namespace-shadows-real-package`)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    92.60s (0:01:32)
- Tier A attempts: 1

## Files changed
**tt-forge-models:**
- `babylm_git/pytorch/requirements.txt` (new)
- `babylm_git/pytorch/loader.py`

**tt-xla:**
- `python_package/tt_torch/backend/passes.py`
- `python_package/tt_torch/backend/backend.py`
- `tests/runner/utils/dynamic_loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 507551a04d75b3e671e70e5abbaf15907b457639 |
| tt-forge-models | 2ac014922cc0130a7545830eb1bdfba1a1d0ec18 |
