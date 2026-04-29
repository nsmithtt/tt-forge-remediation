# Remediation Summary: bartowski_thedrummer_fallen_gemma3_27b_v1_gguf-causal_lm-pytorch-27B_V1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_thedrummer_fallen_gemma3_27b_v1_gguf/causal_lm/pytorch-27B_V1_GGUF-single_device-inference]

## Result
XFAIL — 27B model exhausts single-device DRAM (hardware capacity ceiling)

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-gguf-checkpoint-model-to-load-kwarg, aten-slice-oob-negative-start-xla

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_65, 2, -1023, 9223372036854775807), kwargs = {})

(Preceded by: TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause

**Bug 1 (loader):** Several GGUF loaders (gpt_oss_swallow, mradermacher_qwen3_5, etc.) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a version that only accepts `(gguf_path, return_tensors=False)`. In transformers 5.2.0, `from_pretrained` now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. When the dynamic test collector imports any of these loaders during discovery, the patch poisons the global namespace, so even models that never patch it (like `bartowski_thedrummer_fallen_gemma3_27b_v1_gguf`) fail with `TypeError` on load.

**Bug 2 (tt-xla):** Gemma3's `SlidingWindowCache.update` does `full_value_states[:, :, -sliding_window+1:, :]` (= `[:, :, -1023:, :]`). The tokenized input is only 23 tokens, so the slice start -1023 is outside `[-23, 22]`. PyTorch CPU clamps out-of-bounds negative start indices (Python slice semantics), but the XLA/TT backend validates and rejects them during FX graph partitioning. `TorchFunctionMode` intercepts `aten.slice.Tensor` in the FX execution path but not in eager mode.

**Bug 3 (hardware):** After the two fixes above, the forward pass starts but OOMs. The 27B-parameter Gemma3 model, loaded in bfloat16, fills ~99% of device DRAM (allocated: 4225214784 B of 4273390016 B per bank across 8 banks). The remaining 46 MB per bank is fragmented into 13 MB largest-contiguous blocks, too small for a 27.6 MB/bank activation buffer needed during computation.

## Fix

**Fix 1 (loader, tt-forge-models):** Updated `_patched_load_gguf_checkpoint` in 26 affected loader files to accept `model_to_load=None` and forward it to `_orig_load_gguf_checkpoint`. This restores compatibility with transformers 5.2.0's updated `load_gguf_checkpoint` signature.

Branch: `remediation/bartowski_thedrummer_fallen_gemma3_27b_v1_gguf-causal_lm-pytorch-27B_V1_GGUF-single_device-inference` in tt-forge-models.

Files changed: 26 loader files under `third_party/tt_forge_models/*/causal_lm/pytorch/loader.py`.

**Fix 2 (tt-xla):** In `python_package/tt_torch/torch_overrides.py`, added interception of `aten.slice.Tensor` in `TorchFunctionOverride.__torch_function__`. When `start < -size`, the start is clamped to `-size` before dispatching, matching PyTorch CPU slice semantics.

Branch: `remediation/bartowski_thedrummer_fallen_gemma3_27b_v1_gguf-causal_lm-pytorch-27B_V1_GGUF-single_device-inference` in tt-xla.

**Fix 3 (tt-xla test config):** Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` to capture the hardware-class OOM.

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    wormhole (8-bank DRAM, ~32 GB total)
- Duration:    322.05s (0:05:22)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 files in `*/causal_lm/pytorch/loader.py` (Fix 1)
- `tt-xla`: `python_package/tt_torch/torch_overrides.py` (Fix 2)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (Fix 3)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ed1100e0948cd2614bef89b80115879316e81b61 |
| tt-forge-models | 85c388d847e66f796d238a5929cb255482578583 |
