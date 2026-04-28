# Remediation Summary: bert-masked_lm-pytorch-BioBERT_Base_Cased_v1.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-BioBERT_Base_Cased_v1.1-single_device-inference]

## Result
SILICON_PASS — loader load_config() was using AutoConfig instead of BertConfig; fixed to BertConfig

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bert-load-config-autoconfig-missing-model-type

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Unrecognized model in dmis-lab/biobert-base-cased-v1.1. Should have a `model_type` key in its config.json.

At: third_party/tt_forge_models/bert/masked_lm/pytorch/loader.py:229: in load_config
    self.config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)

(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is an unrelated SWIG warning printed at process exit; the true fatal error is the ValueError above.)

## Root cause
The `load_config()` method in `bert/masked_lm/pytorch/loader.py` called `AutoConfig.from_pretrained`, which requires a `model_type` key in the checkpoint's `config.json`. BioBERT (`dmis-lab/biobert-base-cased-v1.1`) predates that convention and omits `model_type`, so `AutoConfig` raises `ValueError`. A partial fix had already landed on the branch (commit `2e73e236ba` "Use BertConfig instead of AutoConfig for BERT masked_lm models") that fixed `load_model()`, but the same change was not applied to `load_config()` and `BertConfig` was not added to the import line.

## Fix
Two changes to `tt_forge_models/bert/masked_lm/pytorch/loader.py`:

1. Added `BertConfig` to the `from transformers import` statement (line 7).
2. Changed `load_config()` to call `BertConfig.from_pretrained(...)` instead of `AutoConfig.from_pretrained(...)` (line 229).

Committed on branch `remediation/bert-masked_lm-pytorch-BioBERT_Base_Cased_v1.1-single_device-inference` in tt-forge-models:
- `d722aac84b` Fix load_config to use BertConfig instead of AutoConfig for BioBERT models missing model_type in config.json
- `8e55437375` Add BertConfig import needed by load_config fix

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    33.57s (wall-clock)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/bert/masked_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8e55437375 |
