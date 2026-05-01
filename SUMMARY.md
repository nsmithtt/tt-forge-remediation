# Remediation Summary: layoutlmv2-pytorch-base-uncased-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[layoutlmv2/pytorch-Base Uncased-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
detectron2-build-isolation-lru-cache-stale

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The CI command was unquoted, producing:
```
ERROR: file or directory not found: Uncased-single_device-inference]
```
(pytest treats the space in "Base Uncased" as an argument separator). When run with proper quoting, the actual failure was:

```
subprocess.CalledProcessError: Command '... pip install --no-input -r .../layoutlmv2/pytorch/requirements.txt' returned non-zero exit status 1.
```
Because `detectron2 @ git+https://github.com/facebookresearch/detectron2.git` failed to build under pip's isolated build environment:
```
ModuleNotFoundError: No module named 'torch'
```
After fixing the build isolation issue, a second failure appeared:
```
ImportError: LayoutLMv2Model requires the detectron2 library but it was not found in your environment.
```

## Root cause
Three interacting loader bugs:

1. **detectron2 build isolation**: `requirements.txt` listed `detectron2 @ git+...` for a direct pip install. detectron2's `setup.py` (and `pyproject.toml`) imports `torch` at build time. pip's build isolation subprocess starts a clean Python environment with no torch, so the build fails immediately. Fix: move detectron2 to `requirements.nodeps.nobuildisolation.txt` so the test framework installs it with `--no-build-isolation --no-deps`.

2. **Missing detectron2 runtime deps**: Installing detectron2 with `--no-deps` skips its runtime dependencies (`fvcore`, `cloudpickle`, `hydra-core`, `omegaconf`). These must be listed in `requirements.txt` for standard pip install.

3. **transformers `is_detectron2_available()` lru_cache stale**: `modeling_layoutlmv2.py` calls `is_detectron2_available()` at module-import time (line 39). The loader's top-level `from transformers import LayoutLMv2Model` triggers this import during test collection, before detectron2 is installed. The `@lru_cache`-decorated function caches `False`. Even after the requirements manager installs detectron2, the cached `False` causes `requires_backends(self, "detectron2")` to raise `ImportError`. Fix: call `is_detectron2_available.cache_clear()` in `load_model()` before instantiating the model.

## Fix
All changes in `tt_forge_models`, branch `remediation/layoutlmv2-pytorch-base-uncased-single-device-inference` (commit `ee361a7629`):

- `layoutlmv2/pytorch/requirements.txt`: replaced `detectron2 @ git+...` with the four runtime deps (`fvcore`, `cloudpickle`, `hydra-core>=1.1`, `omegaconf>=2.1,<2.4`).
- `layoutlmv2/pytorch/requirements.nodeps.nobuildisolation.txt`: new file containing `git+https://github.com/facebookresearch/detectron2.git` — installed with `--no-build-isolation --no-deps` by the test framework.
- `layoutlmv2/pytorch/loader.py`: added `is_detectron2_available.cache_clear()` call in `load_model()` before `LayoutLMv2Model.from_pretrained()`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    198.82s (0:03:18)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/layoutlmv2/pytorch/requirements.txt
- tt_forge_models/layoutlmv2/pytorch/requirements.nodeps.nobuildisolation.txt (new)
- tt_forge_models/layoutlmv2/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 839308184bdaa6c2e64bd5498e19a9aa91d3aaee |
| tt-forge-models | ee361a76291f91d621ad65154dcaa7dac60bdc7c |
