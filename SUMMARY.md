# Remediation Summary: mlx_community_qwen3_5_9b_bf16/image_text_to_text/pytorch-9b_bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_qwen3_5_9b_bf16/image_text_to_text/pytorch-9b_bf16-single_device-inference]

## Result
FAIL — TT PJRT backend cannot execute device→CPU tensor transfers required by Qwen3.5-VL data-dependent control flow (`grid_thw.tolist()`)

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

Full traceback (leading to failure):
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1938: in forward
    outputs = self.model(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1663: in forward
    image_outputs = self.get_image_features(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1546: in get_image_features
    vision_output = self.visual(...)
transformers/models/qwen3_5/modeling_qwen3_5.py:1239: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
transformers/models/qwen3_5/modeling_qwen3_5.py:1162: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Layer**: Compiler frontend (tt-xla) — TT PJRT plugin (`pjrt_plugin_tt.so`)

`Qwen3_5VisionModel.fast_pos_embed_interpolate` (transformers `modeling_qwen3_5.py:1162`) calls
`grid_thw.tolist()` to iterate over per-image grid dimensions (temporal, height, width) for
positional-embedding interpolation. At inference time `grid_thw` is a tensor on the TT device.
Calling `.tolist()` on a TT device tensor requires a device→host data transfer via the PJRT
interface. The TT PJRT plugin does not implement this transfer path: the call reaches
`tt_torch/torch_overrides.py:34` via `__torch_function__`, which dispatches the native
`.tolist()`, which in turn fails with `INTERNAL: Error code: 13`.

This is the same bug previously diagnosed for `berkerdooo_qwen3_5_27b_nvfp4`
(report branch `report/berkerdooo-qwen35-27b-nvfp4-tolist`). That report confirmed that even
an explicit `.cpu()` call on `grid_thw` before passing to the vision function fails with the
same error code (the synchronisation in `torch_xla.sync` inside
`dynamo_bridge.extract_compiled_graph_helper` also returns `INTERNAL: Error code: 13`).

## Fix
The fix requires changes in **tt-xla** (compiler frontend):

1. Implement `TransferLiteralFromDevice` / `CopyDeviceMemoryToHost` in `pjrt_plugin_tt.so` so
   that `tensor.cpu()`, `tensor.item()`, and `tensor.tolist()` on TT device tensors succeed.
2. Alternatively, implement CPU-fallback detection in `tt_torch/backend/backend.py` so that
   tensors involved in Python-level control flow are transparently transferred to host before
   the compiled region requires them.

No loader-layer fix is possible without forbidden patterns (CPU offload of the visual encoder
or switching to text-only inputs).

## Tier B justification
**Indicator**: `internal-error-unknown-mechanism`

The TT PJRT plugin has no implemented device→host tensor-read path. Adding it requires new
PJRT transfer infrastructure (`TransferLiteralFromDevice`), not a scoped one- or two-file
patch. The attempted fix in the berkerdooo remediation confirmed that even rerouting via
`.cpu()` fails at the PJRT synchronisation step — the root cause is deeper than a single
function fix.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 402.45s (0:06:42)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
