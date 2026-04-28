# Remediation Summary: dpt-pytorch-large-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dpt/pytorch-Large-single_device-inference]

## Result
SILICON_PASS — loader fixes resolved both the spacy namespace collision and the DPTImageProcessor breaking change

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-use-fast-default

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

Additionally, the test was failing with:
    AttributeError: module 'spacy' has no attribute 'Language'
from `datasets` dill serialization inside `load_dataset("huggingface/cats-image")`.

## Root cause
Two loader-layer bugs:

**Bug 1 (huspacy, blocking):** `huspacy/pytorch/loader.py` imported `spacy` at module
level. The `dynamic_loader` adds `tt_forge_models/` to `sys.path` during test discovery;
this makes the `tt_forge_models/spacy/` directory visible as a top-level namespace
package. When `huspacy`'s module-level `import spacy` ran, Python resolved it to that
namespace package (no `Language` attribute) instead of the real spaCy library.
Later, the `datasets` dill serializer checks `if "spacy" in sys.modules` → True, then
attempts `spacy.Language` → `AttributeError`.

**Bug 2 (dpt, transformers 5.x):** transformers 5.2.0 changed the default for
`AutoImageProcessor.from_pretrained` to load the fast processor variant when one is
available, even if the checkpoint was saved with the slow processor. For `Intel/dpt-large`
this means `DPTImageProcessorFast` is loaded instead of `DPTImageProcessor`, which can
produce different outputs.

## Fix
**Fix 1:** Moved `import spacy` from module level inside the `_load_nlp()` method of
`huspacy/pytorch/loader.py`. This defers the import until the model is actually used,
preventing the namespace-package collision from affecting other loaders during discovery.

**Fix 2:** Added `use_fast=False` to `AutoImageProcessor.from_pretrained(pretrained_model_name)`
in `dpt/pytorch/loader.py`'s `_load_processor()` method, keeping the slow
`DPTImageProcessor` as the checkpoint expects.

Both fixes committed to `tt_forge_models` branch
`remediation/dpt-pytorch-large-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    142.65s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/dpt/pytorch/loader.py` — added `use_fast=False` to `AutoImageProcessor.from_pretrained`
- `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py` — moved `import spacy` inside `_load_nlp()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b8a4c138681974af839f7b2cb9554802a03f0742 |
| tt-forge-models | 6792125eeffc58b2316bad1a215f4be31d49ffd6 |
