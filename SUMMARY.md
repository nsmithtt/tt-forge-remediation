# Remediation Summary: jackdaw_gguf-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jackdaw_gguf/causal_lm/pytorch-30B_A3B_i1_GGUF-single_device-inference]

## Result
XFAIL — 30B Qwen3 MoE model dequantizes to ~60 GB bfloat16, exceeding single-device DRAM (p150b: 24 GB)

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
Fatal Python error: Segmentation fault

## Root cause
Two loader-layer bugs obscured the hardware-capacity ceiling:

1. **Segfault (primary)**: `Qwen3MoeExperts.forward()` dispatches experts via
   a Python for-loop over a dynamically-sized `expert_hit` tensor. XLA cannot
   statically trace a loop whose iteration count depends on a runtime nonzero()
   result; this caused a segfault during graph partition probing. Fix: set
   `model.config._experts_implementation = "batched_mm"` after loading, which
   replaces the dynamic loop with static scatter/gather/matmul ops.

2. **TypeError (secondary)**: 26 GGUF loader wrappers in tt_forge_models used a
   narrow `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   signature. Transformers 5.2.0 added a `model_to_load=` kwarg to
   `load_gguf_checkpoint`. When any of those 26 loaders had monkey-patched the
   global symbol before jackdaw's `from_pretrained` ran, the call raised
   `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument
   'model_to_load'`. Fix: updated all 26 wrappers to forward `**kwargs`.

After both loader fixes, the model loads and compiles successfully but fails at
`_run_cached_graph` with `INTERNAL: Error code: 13` (RESOURCE_EXHAUSTED / OOM).
The Jackdaw-30B-A3B is a Qwen3 MoE fine-tune: 30B total parameters × 2 bytes
(bfloat16 dequantized at load time) ≈ 60 GB, which far exceeds the 24 GB DRAM
of a p150b and the 12 GB of an n150.

## Fix
1. `jackdaw_gguf/causal_lm/pytorch/loader.py` in `tt-forge-models` repo
   (branch `remediation/jackdaw-gguf-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference`):
   Added `model.config._experts_implementation = "batched_mm"` after `from_pretrained`.

2. 26 GGUF loader wrappers in `tt-forge-models` (same branch): updated
   `_patched_load_gguf_checkpoint` signature from
   `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)`
   and added `**kwargs` to the internal `_orig_load_gguf_checkpoint(...)` call.

3. `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
   in `tt-xla` repo (same branch): added `KNOWN_FAILURE_XFAIL` entry for
   `jackdaw_gguf/causal_lm/pytorch-30B_A3B_i1_GGUF-single_device-inference`
   with reason "30B MoE model dequantizes to ~60 GB bfloat16 at load time,
   exceeding single-device DRAM (p150b: 24 GB)".

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — device OOM after segfault fixed)
- Hardware:    n150
- Duration:    1604.49s (0:26:44)
- Tier A attempts: N/A

## Files changed
- `jackdaw_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- 26 × `*/causal_lm/pytorch/loader.py` — `_patched_load_gguf_checkpoint` kwargs fix (tt-forge-models)
- `.gitignore` — exclude venv/ (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 88c536912487225c9f9d0b7ce9c0bb2d032a315b |
| tt-forge-models | 2ca4b1b023ecde17f2219fcc7973d5eb9d7ebe44 |
