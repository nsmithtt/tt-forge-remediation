# Remediation Summary: mambavision-image_classification-pytorch-MambaVision-T-1K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mambavision/image_classification/pytorch-MambaVision-T-1K-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed; model compiles and runs on TT hardware

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ModuleNotFoundError: No module named 'mamba_ssm.ops.selective_scan_interface'; 'mamba_ssm.ops' is not a package
```
(original crash, shown as Extension modules dump in CI)

## Root cause
Three separate loader bugs in nvidia/MambaVision-T-1K with transformers 5.x:

**Bug 1 - mamba_ssm CUDA import:** modeling_mambavision.py does a top-level
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn. The real
mamba-ssm package requires CUDA to build. An existing mamba_ssm.py file in
tt-metal (a plain module, not a package) meant mamba_ssm.ops failed to import.

**Bug 2 - meta tensor init:** Transformers 5.x get_init_context() wraps model
construction in torch.device("meta"). MambaVision.__init__ calls
torch.linspace(0, drop_path_rate, sum(depths)).item() to build drop-path rates;
.item() cannot be called on meta tensors.

**Bug 3 - missing all_tied_weights_keys:** MambaVisionModelForImageClassification
does not call self.post_init(), so transformers 5.x _finalize_model_loading fails
with AttributeError on all_tied_weights_keys.

An additional pre-existing bug (spacy namespace shadowing) was also fixed in
dynamic_loader: sys.path.insert(models_root) caused tt_forge_models/spacy/ to
shadow the real spaCy, breaking datasets._dill during load_inputs.

## Fix
**Fix 1 (loader):** Install a pure-PyTorch stub for mamba_ssm.ops.selective_scan_interface
before the model import. The stub implements the SSM reference recurrence in PyTorch.

**Fix 2 (loader):** Temporarily patch PreTrainedModel.get_init_context to strip
torch.device context managers so torch.linspace().item() works during __init__.

**Fix 3 (loader):** Temporarily patch _adjust_tied_keys_with_tied_pointers to
lazily initialize all_tied_weights_keys = {} if the attribute is missing.

**Fix 4 (dynamic_loader):** Cherry-picked commit 8cc0addbe to remove
sys.path.insert(models_root) that shadowed the real spaCy package.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    361s (6:01)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/mambavision/image_classification/pytorch/loader.py
- tt-xla/tests/runner/utils/dynamic_loader.py (cherry-pick)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 52422438c |
| tt-forge-models | ea02698c0f |
