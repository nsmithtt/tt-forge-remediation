# Remediation Summary: gliese_qwen3_5/image_to_text/pytorch-9b_abliterated_caption-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gliese_qwen3_5/image_to_text/pytorch-9b_abliterated_caption-single_device-inference]

## Result
FAIL — TT PJRT backend cannot execute device→CPU tensor transfers required by Qwen3.5-VL data-dependent control flow

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Full traceback at point of failure:
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1239: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
transformers/models/qwen3_5/modeling_qwen3_5.py:1162: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Layer**: tt-xla (compiler frontend / PJRT plugin)

The Qwen3.5-9B-VL model's visual encoder requires data-dependent Python control flow inside the compiled forward pass. `Qwen3_5VisionModel.fast_pos_embed_interpolate` (transformers `modeling_qwen3_5.py:1162`) calls `grid_thw.tolist()` to iterate over per-image grid dimensions (`t`, `h`, `w`) for `torch.linspace` calls, Python loop bounds, and tensor split sizes. At runtime, `grid_thw` is an `int64` tensor that has been moved to TT device (by the test harness's `to_device` in `tests/infra/runners/torch_device_runner.py`). The `__torch_function__` override in `python_package/tt_torch/torch_overrides.py:34` intercepts the `.tolist()` call and delegates to the TT PJRT backend, which cannot complete the device→host data transfer and fails with `INTERNAL: Error code: 13`.

This is the same root cause already documented in report `berkerdooo-qwen35-27b-nvfp4-tolist` for the 27B variant. That report also attempted a loader-layer fix (patching `fast_pos_embed_interpolate` to call `.cpu()` on `grid_thw` before `.tolist()`), but the explicit `.cpu()` transfer equally fails via `torch_xla.sync` at `_xla_step_marker`. The bug is not in the model loader; it is in the TT PJRT plugin's inability to transfer tensor data from device to host memory.

## Fix
The fix must be made in the **compiler frontend layer (tt-xla)**, specifically in the TT PJRT plugin:

1. **Implement device→host tensor data transfer** in `pjrt_plugin_tt.so`. The PJRT interface specifies `TransferLiteralFromDevice` / `CopyDeviceMemoryToHost`; the TT implementation must handle these paths so that `tensor.cpu()`, `tensor.item()`, and `tensor.tolist()` on TT device tensors succeed.

2. **Alternatively**: implement CPU-fallback detection in `tt_torch/backend/backend.py` so that tensors used for Python-level control flow (via `.tolist()` / `.item()` / `.cpu()` usage) are transparently fetched from device before the compiled region consumes them.

No loader-layer (tt_forge_models) workaround is possible without forbidden patterns (moving the visual encoder or RoPE computation to CPU, or bypassing the vision path with text-only inputs).

## Tier B justification
**Indicator**: new-infrastructure

Fixing this requires implementing PJRT device→host tensor transfer paths in the TT PJRT plugin (`pjrt_plugin_tt.so`). The prior report (`berkerdooo-qwen35-27b-nvfp4-tolist`) confirmed that even an explicit `.cpu()` call at the loader layer fails at `torch_xla.sync`, proving the underlying transfer infrastructure is absent. This is not a scoped pattern or threshold fix — it requires new PJRT transport infrastructure.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 432.66s (0:07:12)
- Tier A attempts: N/A

## Files changed
None — Tier B bug, no fix attempted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
