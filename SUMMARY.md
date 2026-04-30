# Remediation Summary: gemma_scope_transcoders-pytorch-gemma-scope-transcoders-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_scope_transcoders/pytorch-gemma-scope-transcoders-single_device-inference]

## Result
FAIL — loader fixes applied but silicon verification blocked by gated-model 403 (HF token lacks access to google/gemma-2-2b)

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
circuit-tracer-missing-requirements-and-hf-hub-compat

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   TypeError: transformers.models.auto.auto_factory._BaseAutoModelClass.from_pretrained() got multiple values for keyword argument 'token'
```

(Locally reproduced first as `ModuleNotFoundError: No module named 'circuit_tracer'`, then as
`ImportError: cannot import name 'HF_HUB_ENABLE_HF_TRANSFER' from 'huggingface_hub.constants'` after
circuit_tracer was manually installed. After applying the loader fixes both errors are resolved and the
test proceeds to `OSError: 403 gated repo` for `google/gemma-2-2b`, which is a local HF-token access
issue, not a code bug.)

## Root cause
The loader (`gemma_scope_transcoders/pytorch/loader.py`) imports `circuit_tracer` but:

1. **Missing `requirements.txt`**: No requirements file existed, so `circuit_tracer` was never
   installed by the test runner's `RequirementsManager`. This caused `ModuleNotFoundError` locally
   and explains why CI hit a stale/wrong `transformer_lens` version that had the `token`
   duplication bug.

2. **`HF_HUB_ENABLE_HF_TRANSFER` removed from `huggingface_hub>=0.22`**: `circuit_tracer 0.5.0`
   imports this constant from `huggingface_hub.constants` at module level. With our project's
   `huggingface_hub==1.12.2` the constant does not exist, causing `ImportError`.

3. **`circuit-tracer`'s strict transformers pin (`<=4.57.3`) conflicts with project env (`5.2.0`)**:
   Using `requirements.nodeps.txt` for the `circuit-tracer` package itself avoids downgrading
   transformers, while `requirements.txt` pulls in `transformer-lens` and `nnsight` which are
   compatible with transformers 5.x. The monkey-patch in the loader injects
   `HF_HUB_ENABLE_HF_TRANSFER` before importing `circuit_tracer` so the constant exists at
   import time regardless of which `huggingface_hub` version is active.

The CI `token duplicate` error was caused by the missing requirements: without `requirements.txt`,
CI was running with a stale/incompatible `transformer_lens` install in the base venv that had the
`token` bug. Installing `circuit-tracer` properly (via the new requirements files) pins
`transformer_lens>=2.16.0`, and `transformer_lens 3.0.0` does not duplicate the `token` argument.

## Fix
All changes are in `tt_forge_models` (`gemma_scope_transcoders/pytorch/`):

**`requirements.txt`** (new file):
```
transformer-lens>=2.16.0
nnsight>=0.6.0
```
Installs the `circuit_tracer` dependencies that are compatible with the project's `transformers 5.x`
environment. These are pulled in via the normal `RequirementsManager` flow.

**`requirements.nodeps.txt`** (new file):
```
circuit-tracer
```
Installs the `circuit-tracer` package itself with `--no-deps`, avoiding the conflicting
`transformers<=4.57.3` and `huggingface-hub<1.0.0` dependency pins.

**`loader.py`** (patched `load_model`):
Before `from circuit_tracer import ReplacementModel`, injects `HF_HUB_ENABLE_HF_TRANSFER` into
`huggingface_hub.constants` if the attribute is absent (removed in `huggingface_hub>=0.22`).

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    22.63s (failed at HF 403 gated-model access, not a code error)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gemma_scope_transcoders/pytorch/requirements.txt` (new)
- `tt_forge_models/gemma_scope_transcoders/pytorch/requirements.nodeps.txt` (new)
- `tt_forge_models/gemma_scope_transcoders/pytorch/loader.py` (monkey-patch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 60ed76e59dba8aa5936c4bc2268d3f197ff580bb |
| tt-forge-models | aa6ea33bbef79144728c34f17b6d15ab08034a47 |
