# Remediation Summary: dummy_tokenizer_fast-masked_lm-pytorch-Dummy_Tokenizer_Fast-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dummy_tokenizer_fast/masked_lm/pytorch-Dummy_Tokenizer_Fast-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-from-pretrained-on-weights-free-repo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: robot-test/dummy-tokenizer-fast-with-model-config does not appear to have a file named pytorch_model.bin or model.safetensors.

(The failure surfaces locally as `ModuleNotFoundError: No module named 'infra'` when tests/ is not in PYTHONPATH; with PYTHONPATH=tests the real error is the OSError above. The CI failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is the last printed line from the SWIG import warning, not the root cause.)

## Root cause
The HuggingFace repo `robot-test/dummy-tokenizer-fast-with-model-config` is a dummy tokenizer test repo that ships a `config.json` and tokenizer files but **no model weights** (`pytorch_model.bin` or `model.safetensors`). The loader called `AutoModelForMaskedLM.from_pretrained(pretrained_model_name)`, which requires actual weights and raises `OSError` when none are found. Since this is an ALBERT-based dummy model intended purely for tokenizer testing, the correct approach is to initialise the model randomly from the available config using `from_config`.

## Fix
`dummy_tokenizer_fast/masked_lm/pytorch/loader.py` in `tt-forge-models`:
- Removed the unused `pretrained_model_name` local variable.
- Replaced `AutoModelForMaskedLM.from_pretrained(pretrained_model_name, **model_kwargs)` with `AutoModelForMaskedLM.from_config(self.config, **model_kwargs)`.

Branch: `remediation/dummy_tokenizer_fast-masked_lm-pytorch-Dummy_Tokenizer_Fast-single_device-inference` in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    61.70s (0:01:01)
- Tier A attempts: N/A

## Files changed
- `dummy_tokenizer_fast/masked_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2672b24d0e2819980a8751dd664f7c76a096618b |
| tt-forge-models | 58ee498d1c61da140ea83eff6c8a42f339f5a773 |
