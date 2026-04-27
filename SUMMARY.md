# Remediation Summary: berkerdooo_qwen3_5_27b_nvfp4/image_to_text/pytorch-27B_NVFP4-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[berkerdooo_qwen3_5_27b_nvfp4/image_to_text/pytorch-27B_NVFP4-single_device-inference]

## Result
FAIL — TT PJRT backend cannot execute device→CPU tensor transfers required by Qwen3.5-VL data-dependent control flow

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

**Original traceback** (without fix, tt_forge_models @ 0f7b7343):
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1162: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

**After attempted fix** (tt_forge_models @ c198d748 — patch calls `.cpu()` on grid_thw before passing to vision functions):
```
third_party/tt_forge_models/.../loader.py:31: in _patched_fast_pos_embed
    def _patched_fast_pos_embed(self, grid_thw):
torch/_dynamo/eval_frame.py:1044: in _fn
tt_torch/backend/backend.py:225: in __call__
    return self._call_experimental_compile(*args)
tt_torch/backend/backend.py:215: in _call_experimental_compile
    self.compiled_graph = bridge.extract_compiled_graph(...)
torch_xla/_dynamo/dynamo_bridge.py:826: in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
torch_xla/torch_xla.py:87: in sync
    torch_xla._XLAC._xla_step_marker(...)
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Layer**: Compiler frontend (tt-xla) — TT PJRT plugin (`pjrt_plugin_tt.so`)

The Qwen3.5-27B VL model's visual encoder uses data-dependent Python control flow inside its compiled forward pass:

- `Qwen3_5VisionModel.fast_pos_embed_interpolate` calls `grid_thw.tolist()` (line 1162) to iterate over per-image grid dimensions
- `Qwen3_5VisionModel.rot_pos_emb` calls `grid_thw.tolist()` (line 1123)
- `Qwen3_5Model.get_rope_index` calls `image_grid_thw.tolist()` and `input_ids.tolist()` (lines 1440, 1467)

All three functions are inside the `torch.compile`d forward pass. When `grid_thw` is a tensor on the TT device, `.tolist()` requires a device→CPU data transfer. The TT PJRT plugin cannot execute this transfer: both the direct `.tolist()` call (via `__torch_function__`) and an explicit `.cpu()` call (which queues a device→host copy that is then synchronised by `torch_xla.sync`) fail with `INTERNAL: Error code: 13` at `_xla_step_marker`.

The attempted fix in `remediation/berkerdooo-qwen35-27b-nvfp4-tolist-patch` wraps these functions to call `.cpu()` on `grid_thw` before dispatching. This moves the `.tolist()` failure to `torch_xla.sync` in `dynamo_bridge.extract_compiled_graph_helper` — same error code, different call site. The underlying limitation is identical: the TT XLA backend cannot read tensor data back from device to host memory.

## Fix
The fix requires changes in the **compiler frontend layer (tt-xla)**, specifically in the TT PJRT plugin:

1. **Implement device→host tensor data transfer** in `pjrt_plugin_tt.so`. The PJRT interface specifies `TransferLiteralFromDevice` / `CopyDeviceMemoryToHost`; the TT implementation needs to handle these paths so that `tensor.cpu()`, `tensor.item()`, and `tensor.tolist()` on TT device tensors succeed.

2. **Alternatively**: implement CPU-fallback detection in `tt_torch/backend/backend.py` so that tensors needed for Python-level control flow (detected via `tolist()` / `item()` / `.cpu()` usage patterns) are transparently fetched from device before the compiled region needs them.

No loader-layer (tt_forge_models) workaround is possible without using forbidden patterns (moving the visual encoder or RoPE computation to CPU, or switching to text-only inputs).

## Verification
- **Without fix**: pytest FAILED in 1327.76s (0:22:07) — `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` at `grid_thw.tolist()` in `fast_pos_embed_interpolate`
- **With attempted loader fix** (`remediation/berkerdooo-qwen35-27b-nvfp4-tolist-patch`): pytest FAILED in 1176.57s (0:19:36) — same error code at `torch_xla.sync` in `extract_compiled_graph_helper`
- Hardware: n150

## Files changed
None (loader fix attempted but insufficient; compiler-stack bug requires tt-xla changes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
