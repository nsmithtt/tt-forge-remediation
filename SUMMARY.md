# Remediation Summary: plip-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[plip/pytorch-Base-single_device-inference]

## Result
SILICON_PASS

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
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

A secondary failure also surfaced locally: `AttributeError: module 'spacy' has no attribute 'Language'` when `load_dataset("huggingface/cats-image")` is called. This is caused by `huspacy/pytorch/loader.py` having a module-level `import spacy` that installs the stub `tt_forge_models/spacy/` namespace package into `sys.modules` at pytest collection time, shadowing the real spacy library.

## Root cause
Two loader-layer bugs:

1. **CLIPImageProcessor fast-processor default** — `CLIPProcessor.from_pretrained` in transformers 5.x now loads `CLIPImageProcessor` as a fast processor by default. This is a breaking change that produces a warning treated as the failure indicator. Fix: pass `use_fast=False` to `CLIPProcessor.from_pretrained`.

2. **spacy namespace collision via huspacy** — `tt_forge_models/` is added to `sys.path` by the dynamic loader. The `tt_forge_models/spacy/` subdirectory is a namespace package that shadows the real `spacy` library. `huspacy/pytorch/loader.py` had a module-level `import spacy`, which installs the stub into `sys.modules` during pytest collection. Any subsequent call to `datasets.load_dataset` then hits `issubclass(obj_type, spacy.Language)` in `datasets/_dill.py` and raises `AttributeError`. The PLIP loader called `load_dataset("huggingface/cats-image")` to obtain a sample image.

## Fix
Changes in `tt-xla/third_party/tt_forge_models` on branch `remediation/plip-pytorch-Base-single_device-inference`:

- `plip/pytorch/loader.py`: Added `use_fast=False` to `CLIPProcessor.from_pretrained`; replaced `load_dataset("huggingface/cats-image")` with `PIL.Image.new("RGB", (224, 224))` to eliminate the fragile datasets dependency for a dummy input image.
- `huspacy/pytorch/loader.py`: Moved `import spacy` from module level into `_load_nlp()` (lazy import) to prevent the namespace collision from propagating to other tests during collection.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    77.88s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/plip/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 42e4cd570a07b389a3b0169a4571ac9a18a257f2 |
