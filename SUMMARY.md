# Remediation Summary: ace_step-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ace_step/pytorch-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
minicpm-nn-module-getattr-global-patch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise RuntimeError("Tensor.item() cannot be called on meta tensors")
```

Secondary failure (exposed after ace-step package pip install downgrades transformers):
```
torch._dynamo.exc.InternalTorchDynamoError: ImportError: cannot import name 'ALL_PARALLEL_STYLES' from 'transformers.integrations.tensor_parallel'
```

## Root cause
Two loader-layer bugs combined to produce the failure.

**Bug 1 (ace_step loader):** The original loader used `AutoModel.from_pretrained(..., trust_remote_code=True)` with transformers 5.x. In transformers 5.x, `PreTrainedModel.get_init_context` always initializes on a `meta` device, so any `Tensor.item()` call inside `ACEStepTransformer2DModel.__init__` raises `RuntimeError: Tensor.item() cannot be called on meta tensors`. The fix (already on the remediation branch) switches to `ACEStepTransformer2DModel.from_pretrained(...)` from the `ace-step` package directly, bypassing transformers' meta-device init path.

**Bug 2 (minicpm loaders — cross-test contamination):** Five minicpm loaders (`minicpmv_2_6`, `minicpm_o_2_6`, `minicpm_o_4_5`, `minicpm_v_2`, `minicpm_v_2_6_int4`) applied `nn.Module.__getattr__ = patched_getattr` at **module-import time** (i.e., during pytest collection), permanently replacing the global `nn.Module.__getattr__` for the entire process. When the ace-step pip install later downgraded transformers from 5.2.0 to 4.50.0, `RequirementsManager._purge_stale_modules()` cleared all `transformers.*` entries from `sys.modules`. When Torch Dynamo subsequently traced `ACEStepTransformer2DModel.forward`, it tried to inline the still-active `patched_getattr` closure. To do so, Dynamo called `importlib.import_module('tt_forge_models.minicpmv_2_6.pytorch.loader')`, which re-executed the module code against transformers 4.50.0 — which lacks `ALL_PARALLEL_STYLES` — producing an `ImportError` wrapped in `InternalTorchDynamoError`.

## Fix
All changes are in `tt-xla/third_party/tt_forge_models` (tt-forge-models repo).

**ace_step loader fixes** (pre-existing on remediation branch, commit `b07ec7a`):
- `ace_step/pytorch/loader.py`: Replaced `AutoModel.from_pretrained(..., trust_remote_code=True)` with `ACEStepTransformer2DModel.from_pretrained(model_name, subfolder="ace_step_transformer")` from the `ace-step` package. Fixed model name (`ACE-Step/ACE-Step-v1-3.5B`), inputs (correct tensor shapes and dtypes for the ACEStep forward signature), and added `unpack_forward_output`.
- `ace_step/pytorch/requirements.txt`: Fixed package name from `acestep @` to `ace-step @`.

**minicpm loader fixes** (commit `59af657a91`):
- `minicpmv_2_6/pytorch/loader.py`: Removed module-level `nn.Module.__getattr__` patch; moved it inside `load_model()` with try/finally to restore original after `AutoModel.from_pretrained` returns.
- `minicpm_o_2_6/pytorch/loader.py`: Same fix.
- `minicpm_o_4_5/pytorch/loader.py`: Same fix.
- `minicpm_v_2/pytorch/loader.py`: Same fix.
- `minicpm_v_2_6_int4/pytorch/loader.py`: Same fix.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    574.03s (0:09:34)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/ace_step/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/ace_step/pytorch/requirements.txt`
- `tt-xla/third_party/tt_forge_models/minicpmv_2_6/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_o_2_6/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_o_4_5/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_v_2/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_v_2_6_int4/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e6dd7a02ed6eb5d1a52879883a8399568c659711 |
| tt-forge-models | 59af657a91c642d4d2ce7cc40f03c6e039399c06 |
