# Remediation Summary: mobileclip2-pytorch-S2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobileclip2/pytorch-S2-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
open-clip-get-tokenizer-unregistered-model-name

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AssertionError: No valid model config found for MobileCLIP2-S2.
```
in `open_clip/factory.py:114` when `get_tokenizer("MobileCLIP2-S2")` is called.

After fixing the tokenizer name, a second bug appears:
```
AttributeError: module 'spacy' has no attribute 'Language'
```
in `datasets/utils/_dill.py:42` when `load_dataset("huggingface/cats-image")` is called.

## Root cause
Three loader bugs:

1. **get_tokenizer name mismatch**: open_clip 2.32.0 does not register "MobileCLIP2-S*" names in its local model config registry. The loader called `get_tokenizer("MobileCLIP2-S2")` which hits the `assert config is not None` guard in `open_clip/factory.py`. Using the `hf-hub:` prefix would reach HuggingFace to fetch the config, but that triggers a second bug.

2. **HFTokenizer.batch_encode_plus removed in transformers 5.x**: `open_clip.get_tokenizer("hf-hub:timm/MobileCLIP2-S2-OpenCLIP")` returns an `HFTokenizer` whose `__call__` method calls `self.tokenizer.batch_encode_plus(...)`. In transformers 5.x, `CLIPTokenizer` no longer inherits `batch_encode_plus`. The fix bypasses open_clip's `HFTokenizer` entirely by loading the tokenizer directly via `AutoTokenizer.from_pretrained` and wrapping it in a small callable (`_CLIPTokenizerWrapper`) that returns token IDs as a plain tensor.

3. **spacy namespace pollution crashes load_dataset**: `tt_forge_models/spacy/` is a namespace package that pollutes `sys.modules['spacy']` with an incomplete object. When `datasets` dill-pickles the data files config, it checks `issubclass(obj_type, spacy.Language)` which raises `AttributeError`. Fixed by replacing `load_dataset("huggingface/cats-image")` with `get_file` + `PIL.Image.open` using the COCO val2017 image URL.

## Fix
Changes in `tt_forge_models/mobileclip2/pytorch/loader.py`:

1. Replaced `_TOKENIZER_NAME` dict (mapping to bare model name strings) with `_TOKENIZER_HF_REPO` dict mapping to HuggingFace repo IDs.

2. Added `_CLIPTokenizerWrapper` class that wraps `AutoTokenizer.from_pretrained` to produce a callable matching the open_clip tokenizer interface (returns `torch.Tensor` of shape `[batch, 77]`). This bypasses both the missing registry entry and the `batch_encode_plus` removal.

3. Replaced `get_tokenizer(_TOKENIZER_NAME[...])` calls with `_CLIPTokenizerWrapper(_TOKENIZER_HF_REPO[...])` in both `load_model` and `load_inputs`.

4. Replaced `load_dataset("huggingface/cats-image")` in `load_inputs` with `get_file("http://images.cocodataset.org/val2017/000000039769.jpg")` + `PIL.Image.open`, avoiding the spacy namespace collision.

Repo: `tt-forge-models`
Branch: `remediation/mobileclip2-pytorch-S2-single_device-inference`
File: `mobileclip2/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    236.92s (0:03:56)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/mobileclip2/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d0a35d2f6e3afae9063cbb1600e4165d42020de3 |
| tt-forge-models | 869f3baa1242f242e125289626697834508302bc |
