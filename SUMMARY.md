# Remediation Summary: accent_id_commonaccent_ecapa-pytorch-CommonAccent_ECAPA-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[accent_id_commonaccent_ecapa/pytorch-CommonAccent_ECAPA-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
speechbrain-missing-requirements-and-bf16-incompatible

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The actual root failure was `ModuleNotFoundError: No module named 'speechbrain'`, which manifested with a different error when speechbrain was partially installed due to torchaudio version conflict.)

## Root cause
Two loader-layer bugs:

1. **Missing requirements.txt**: `speechbrain` was not declared as a dependency.
   The loader calls `from speechbrain.inference.speaker import EncoderClassifier`
   but there was no `requirements.txt` to install it, causing
   `ModuleNotFoundError: No module named 'speechbrain'`.

2. **bfloat16 incompatibility**: `speechbrain.lobes.models.ECAPA_TDNN.AttentiveStatisticsPooling.forward`
   has a hardcoded `.float()` call (line 284 in ECAPA_TDNN.py) that creates
   float32 intermediate tensors (`total = mask.sum(dim=2, keepdim=True).float()`).
   These propagate through `_compute_statistics` to make `mean` and `std` float32,
   which are then concatenated with the bfloat16 input `x`, yielding a float32 `attn`.
   When `attn` is passed to `self.tdnn` (which has bfloat16 bias), PyTorch raises:
   `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same`.
   Fix: remove `dtype_override` from `load_model` and `load_inputs` so the model
   stays in float32 (its native dtype).

3. **torchaudio CPU wheel pinning**: Without `--extra-index-url https://download.pytorch.org/whl/cpu`
   and an explicit `torchaudio==2.9.0` pin, `pip install speechbrain` fetches
   the CUDA-linked torchaudio from PyPI, which also changes torch. This causes
   `RequirementsManager._purge_stale_modules()` to evict `torch` from
   `sys.modules`; the next `import torch` (from speechbrain) re-executes
   `torch/__init__.py` and hits `RuntimeError: function '_has_torch_function'
   already has a docstring` because the C extension was already initialized.

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch
`remediation/accent_id_commonaccent_ecapa-pytorch-CommonAccent_ECAPA-single_device-inference`:

1. **`accent_id_commonaccent_ecapa/pytorch/requirements.txt`** (new file):
   Added `--extra-index-url https://download.pytorch.org/whl/cpu`, `speechbrain`,
   and `torchaudio==2.9.0` to install speechbrain and ensure torchaudio is
   fetched from the CPU wheel index (matching the pre-installed `torchaudio==2.9.0+cpu`).

2. **`accent_id_commonaccent_ecapa/pytorch/loader.py`**:
   - Changed `load_model(self, *, dtype_override=None, **kwargs)` to
     `load_model(self, **kwargs)` with `kwargs.pop("dtype_override", None)`.
     Keeps the model in native float32 because speechbrain's
     `AttentiveStatisticsPooling` has a hardcoded `.float()` call incompatible
     with bfloat16 model parameters.
   - Changed `load_inputs(self, dtype_override=None)` to `load_inputs(self)`.
     Float32 inputs match the float32 model (no dtype mismatch).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    82.20s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/accent_id_commonaccent_ecapa/pytorch/requirements.txt` (new)
- `tt-xla/third_party/tt_forge_models/accent_id_commonaccent_ecapa/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0adc52718eb1de1f0da3595543717a931647ac0e |
| tt-forge-models | 97dc4449088db2a6c262c72a4f57db0066216bfd |
