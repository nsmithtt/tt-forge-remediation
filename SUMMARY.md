# Remediation Summary: acoustic_bench-speech_recognition-pytorch-Acoustic_Bench-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[acoustic_bench/speech_recognition/pytorch-Acoustic_Bench-single_device-inference]

## Result
FAIL — all three TypeErrors fixed at loader level; test now fails with OSError because the HuggingFace repo KarthikSivaramaKrishnan/acoustic-bench is private and the environment's HF token is invalid

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
nemo-hfhubmixin-missing-requirements-and-api

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise TypeError(
    `OneLoggerPTLTrainer.save_checkpoint: weights_only must be a supertype of
    `typing.Optional[bool]` but is `<class 'bool'>`
```

This TypeError is raised by the `overrides` library at class-definition time
when importing `nv_one_logger.training_telemetry.integration.pytorch_lightning`,
which is pulled in transitively by `nemo.collections.speechlm2.models`. It fires
because `nv_one_logger_pytorch_lightning_integration` 2.3.1 declared
`OneLoggerPTLTrainer.save_checkpoint` with `weights_only: bool`, but
`lightning>=2.5` changed `Trainer.save_checkpoint` to use `weights_only: Optional[bool]`.
The `@override` decorator's strict signature check then raises TypeError at
import time.

After fixing the lightning version, a second TypeError emerges:
```
TypeError: HFHubMixin._from_pretrained() missing 2 required keyword-only
arguments: 'proxies' and 'resume_download'
```
`huggingface_hub>=1.0` removed these arguments from the internal
`_from_pretrained` call chain, but NeMo's `HFHubMixin._from_pretrained`
still requires them without defaults.

Additionally, the loader was calling `SALM.restore_from()` — the old NeMo 1.x
ModelPT API — but `SALM` in NeMo 2.x inherits from `LightningModule` +
`HFHubMixin`, not `ModelPT`, so `restore_from` is not defined at all.

## Root cause
Three compounding loader bugs:

1. **Missing requirements.txt**: No dependency pins for `nemo-toolkit`,
   `lightning`, or any of nemo's many transitive dependencies. When nemo is
   installed without pinning `lightning<2.5`, the overrides library raises
   a `TypeError` at import time due to `nv_one_logger` 2.3.1's
   `OneLoggerPTLTrainer.save_checkpoint` signature not matching the updated
   `Trainer.save_checkpoint` in lightning 2.5+.

2. **Wrong API call**: `SALM.restore_from()` is the NeMo 1.x ModelPT API.
   `SALM` in NeMo 2.x is a `LightningModule + HFHubMixin`; the correct API is
   `SALM.from_pretrained()` from `PyTorchModelHubMixin`.

3. **NeMo HFHubMixin incompatible with huggingface_hub>=1.0**: NeMo's
   `HFHubMixin._from_pretrained` declares `proxies` and `resume_download` as
   required keyword-only arguments, but `huggingface_hub>=1.0` no longer passes
   them in the `_from_pretrained` call. This causes a second TypeError when
   `SALM.from_pretrained()` is invoked.

## Fix
Three changes, all in the loader (`tt-forge-models` remediation branch):

**Commit 1** (`edab8085f1`): `acoustic_bench: add requirements.txt and fix restore_from to from_pretrained`
- Added `acoustic_bench/speech_recognition/pytorch/requirements.txt` with
  `nemo-toolkit>=2.6.0`, `lightning>=2.2.1,<2.5.0` (to avoid overrides TypeError),
  and all transitive nemo dependencies needed for import.
- Changed `SALM.restore_from(...)` → `SALM.from_pretrained(...)` in `loader.py`.

**Commit 2** (`81c40bdc7d`): `acoustic_bench: patch HFHubMixin for huggingface_hub>=1.0 compatibility`
- Added a `_compat_from_pretrained` wrapper in `load_model()` that supplies
  `proxies=None, resume_download=None` defaults, so NeMo's `HFHubMixin` works
  with both old and new `huggingface_hub`.

After all three fixes, the test no longer raises any TypeError; it fails with
`OSError: KarthikSivaramaKrishnan/acoustic-bench is not a local folder and is
not a valid model identifier` because the HuggingFace model is private and the
test environment's HF token (`hf_VdhavkN...`) is invalid/expired.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `acoustic_bench/speech_recognition/pytorch/requirements.txt` (new)
- `acoustic_bench/speech_recognition/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 81c40bdc7d8d95697e41c8b41e6ce3d76b7e1e57 |
