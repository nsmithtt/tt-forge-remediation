# Remediation Summary: bert_masked_lm-pytorch-Large_Portuguese_Cased-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert/masked_lm/pytorch-Large_Portuguese_Cased-single_device-inference]

## Result
SILICON_PASS — removed undefined ModelVariant enum member reference in BERT masked_lm loader

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-loader-undefined-enum-member-reference

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

The CI `failure_detail` was the last stderr line (a SWIG DeprecationWarning emitted at process
start by an unrelated import). The actual test failure was an `AttributeError` in stdout:

```
AttributeError: 'ModelVariant' object has no attribute 'TOHOKU_NLP_BERT_BASE_JAPANESE_CHAR_V2'
```

## Root cause
In `bert/masked_lm/pytorch/loader.py`, the `load_model` method contained:

```python
if self._variant == ModelVariant.TOHOKU_NLP_BERT_BASE_JAPANESE_CHAR_V2:
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
else:
    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
```

`ModelVariant.TOHOKU_NLP_BERT_BASE_JAPANESE_CHAR_V2` was never defined in the `ModelVariant`
StrEnum, so accessing it raised `AttributeError`. This affected every BERT masked_lm variant
(including `Large_Portuguese_Cased`) because the conditional was evaluated unconditionally on
every `load_model` call.

The CI mis-categorised this as `failure_category: unknown` because `AttributeError` is not in
`failure_patterns.yaml`, and the last non-empty stderr line (the SWIG warning from an unrelated
import) became `failure_detail`.

## Fix
Removed the undefined-variant conditional and the unused `AutoConfig`/`AutoTokenizer` imports
from `bert/masked_lm/pytorch/loader.py` in the tt-forge-models repo.

**Repo**: `git@github.com:tenstorrent/tt-forge-models.git`
**Branch**: `remediation/bert_masked_lm-pytorch-Large_Portuguese_Cased-single_device-inference`
**Commit**: `58b0696f8cdcdf341636ce5d53b568f7eab8a34c`

Files changed:
- `bert/masked_lm/pytorch/loader.py`: removed `if self._variant == ModelVariant.TOHOKU_NLP_BERT_BASE_JAPANESE_CHAR_V2` conditional and the two unused imports (`AutoConfig`, `AutoTokenizer`)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    79.67s (0:01:19)
- Tier A attempts: N/A

## Files changed
- `bert/masked_lm/pytorch/loader.py` (tt-forge-models, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5e9e4c476f25df174d5fe4d14ed00c5c8ac47b6b |
| tt-forge-models | 58b0696f8cdcdf341636ce5d53b568f7eab8a34c |
