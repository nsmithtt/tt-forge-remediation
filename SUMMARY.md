# Remediation Summary: mradermacher_planoo_zirelum_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_planoo_zirelum_gguf/causal_lm/pytorch-Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 30B Qwen3 MoE model exceeds single-device DRAM on all supported hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-30b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Original reported failure: Fatal Python error: Segmentation fault — from Qwen3MoeExperts
dynamic nonzero() expert-dispatch loop that XLA cannot statically trace.)

## Root cause
**Architecture**: `general.architecture = qwen3moe` (Qwen3 MoE 30B, 128×1.8B experts;
base model Katanemo Plano-Orchestrator-30B-A3B).  Total parameters ≈ 30B; after
GGUF Q4_K_M dequantization to BF16 the model requires ≈ 60 GB device memory.
Single-device DRAM limits are n150: 12 GB, p150b: 24 GB — this model cannot fit
on any single TT device.

Two loader bugs were also present that prevented the model from even loading:

1. **transformers 5.2.0 kwarg incompatibility (loader)**: Other co-collected loaders
   install a module-level `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   at pytest collection time. Transformers 5.2.0 added `model_to_load=` to the
   `load_gguf_checkpoint` call in `modeling_utils.py:4016`, causing a TypeError
   when the broken patch is active.

2. **Qwen3MoeExperts segfault (loader)**: `Qwen3MoeExperts.forward()` dispatches
   experts via a Python for-loop over a dynamic `nonzero()` result.  XLA cannot
   statically trace this; the graph-partition probe segfaults.  Setting
   `model.config._experts_implementation = "batched_mm"` routes through a static
   matmul path instead.

After both loader fixes the model loads successfully and attempts to compile on
device, where it immediately fails with OOM (hardware class).

## Fix
**Loader** (`tt_forge_models/mradermacher_planoo_zirelum_gguf/causal_lm/pytorch/loader.py`):

1. Added `_find_real_load_gguf()` helper that walks the `_orig_load_gguf_checkpoint`
   globals chain from the currently-installed patched function back to the real
   transformers `load_gguf_checkpoint` (one that accepts `model_to_load`).

2. In `load_model`, temporarily installs the real function on the three modules
   (`gguf_utils`, `tok_auto`, `config_utils`) for the duration of `from_pretrained`,
   then restores the previous state (so other tests keep their patches).

3. After `from_pretrained`, sets `model.config._experts_implementation = "batched_mm"`.

**Test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):

Added `KNOWN_FAILURE_XFAIL` entry with reason documenting the hardware capacity ceiling.

## Verification
- pytest exit: XFAIL
- Hardware:    n150
- Duration:    106.54s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mradermacher_planoo_zirelum_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b550b586afc2f0960594a2f95098d748487e2e8a |
| tt-forge-models | 65f076436b716e7e9ee8e7651b9b8468394121af |
