# Remediation Summary: hyperclovax-seed_think-pytorch-SEED_Think_32B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hyperclovax/seed_think/pytorch-SEED_Think_32B-single_device-inference]

## Result
XFAIL — HyperCLOVAX-SEED-Think-32B (32B params, ~64 GB BF16) far exceeds n150 DRAM (12 GB); loader bugs fixed but model cannot run on any single TT device

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
transformers-5x-no-init-weights-removed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Primary failure (reproduced): ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils' — the model's remote code (modeling_vlm.py:33) imports this function which was removed in transformers 5.x.

Original ticket failure: "The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`."

## Root cause
Two loader bugs in `hyperclovax/seed_think/pytorch/loader.py`:

1. **no_init_weights removed** (transformers 5.x): The model's custom remote code (`modeling_vlm.py:33`) imports `no_init_weights` directly from `transformers.modeling_utils`. This function was removed in transformers 5.x. The `AutoModelForCausalLM.from_pretrained()` call fails with `ImportError` when executing the remote code.

2. **use_fast default changed** (transformers 5.x): `AutoProcessor.from_pretrained()` was called without `use_fast=False`. In transformers 5.x, `Qwen2VLImageProcessor` is loaded as the fast variant by default, producing a breaking-change warning (and potentially incorrect outputs or errors in older CI runs where this was treated as an error).

Additionally, the model is hardware-class XFAIL: HyperCLOVAX-SEED-Think-32B has 32B parameters, requiring ~64 GB BF16 DRAM, which far exceeds the 12 GB available on a single n150 device.

## Fix
**tt_forge_models** (`hyperclovax/seed_think/pytorch/loader.py`):
- Added `contextlib` import and a `no_init_weights` shim injected into `transformers.modeling_utils` at loader import time, guarded by `hasattr` so it is only applied when the function is absent (i.e., transformers 5.x).
- Added `use_fast=False` to `AutoProcessor.from_pretrained()` to prevent the fast-processor default breaking change.

**tt-xla** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` entry for `hyperclovax/seed_think/pytorch-SEED_Think_32B-single_device-inference` with hardware-capacity reason.

## Verification
- pytest exit: FAIL (hardware capacity; not run after loader fix — 32B model would OOM any single device)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hyperclovax/seed_think/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 4a0f1f87ffd41b62abaf2d1566ef83a433c84c2f |
| tt-forge-models | 6df821d067a1561aefe1c05caec9e49dca78a50f |
