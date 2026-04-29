# Remediation Summary: camembert-sentiment_analysis-pytorch-tblard_tf_allocine-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[camembert/sentiment_analysis/pytorch-tblard_tf_allocine-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-from-tf-removed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: tblard/tf-allocine does not appear to have a file named pytorch_model.bin or model.safetensors.

## Root cause
The loader variant `TBLARD_TF_ALLOCINE` pointed to `tblard/tf-allocine`, a TensorFlow-only HuggingFace model (only `tf_model.h5` weights). It used `from_tf=True` in `from_pretrained()` to load the TF weights. In transformers 5.x, the `from_tf` parameter was removed — it is now silently popped from kwargs (modeling_utils.py line 3886) before `_get_resolved_checkpoint_files()` is called, which then fails because no PyTorch-format weights exist in the repo.

## Fix
Updated `tt_forge_models/camembert/sentiment_analysis/pytorch/loader.py`:
- Changed `pretrained_model_name` for `TBLARD_TF_ALLOCINE` from `tblard/tf-allocine` to `philschmid/pt-tblard-tf-allocine`, which is the community PyTorch conversion of the original model with identical labels (`{'0': 'NEGATIVE', '1': 'POSITIVE'}`) and architecture.
- Removed the now-unnecessary `_FROM_TF_VARIANTS` set and `from_tf=True` code path in `load_model()`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    69.30s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/camembert/sentiment_analysis/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a3e3f57a9778f2844b39ebc6a4727f679f19de4d |
| tt-forge-models | f0a48e3f3e610249511f02492ac61d11650d6e6f |
