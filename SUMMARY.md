# Remediation Summary: medmo_8b-pytorch-8b-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[medmo_8b/pytorch-8b-single_device-inference]

## Result
FAIL — Tier B compiler bug: `grid_thw.tolist()` on TT device tensor in `fast_pos_embed_interpolate` fails with INTERNAL Error code 13 (pjrt-device-to-host-transfer)

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full traceback path:
  transformers/models/qwen3_vl/modeling_qwen3_vl.py:699 in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
  python_package/tt_torch/torch_overrides.py:34 in __torch_function__
    return func(*args, **(kwargs or {}))
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
MedMO-8B uses `Qwen3VLForConditionalGeneration` as its backbone. During the
vision encoder's positional embedding computation, `Qwen3VLVisionModel.forward`
calls `self.fast_pos_embed_interpolate(grid_thw)` at
`modeling_qwen3_vl.py:778`, which in turn calls `grid_thw.tolist()` at line
699. At inference time on TT silicon, `grid_thw` is a device tensor (it was
passed in as part of the image preprocessing outputs). The `tolist()` call
dispatches through `tt_torch/torch_overrides.py:__torch_function__`, which
routes to the TT PJRT backend. The TT PJRT plugin does not support
device-to-host tensor reads of this form and raises `INTERNAL: Error code: 13`.

This is the same class of bug as the previously documented Qwen2VL /
Qwen2.5-VL failure (`pjrt-device-to-host-transfer`), where `grid_thw.tolist()`
in `rot_pos_emb`/`get_window_index` caused identical failures. The Qwen3VL
code path is `fast_pos_embed_interpolate` (a new function in Qwen3VL) rather
than the older `rot_pos_emb`, but the root cause is identical: the TT PJRT
layer cannot transfer a live device tensor to host via Python's `.tolist()`.

All existing Qwen3VL image_to_text models (2B instruct, 2B thinking, 4B
instruct, 4B thinking) are already marked `KNOWN_FAILURE_XFAIL` with
Issue #3184 (`handle->HasValue()` / UNKNOWN_SCALAR). The 8B MedMO-8B variant
hits this earlier code path (`grid_thw.tolist()` in `fast_pos_embed_interpolate`)
and surfaces as `INTERNAL: Error code: 13` instead, but the underlying cause
is the same missing device-to-host transfer infrastructure.

## Fix
Proposed fix (not implemented — Tier B):

The fix lives in `tt-xla`'s PJRT layer. When `.tolist()` or any scalar-read
operation is called on a TT device tensor during eager execution (i.e., outside
a compiled graph), the PJRT plugin should synchronize the device, transfer the
tensor data to host, and return a Python list — exactly as CUDA PJRT does.
This requires implementing a device→host transfer path in the `tt_torch`
bridge, likely in `python_package/tt_torch/torch_overrides.py` or the
underlying `TTPJRTBuffer` implementation in the C++ plugin.

An alternative loader-layer approach — wrapping `grid_thw` in a
`torch._dynamo.disable` region so it stays on CPU — is a forbidden workaround
(CPU offload of model component) and must not be used.

## Tier B justification
cross-cutting — Implementing `.tolist()` / scalar-read support requires changes
to the PJRT buffer transfer path in `tt-xla`'s C++ plugin and Python bridge,
affecting every op that reads a device tensor back to host. This is new
infrastructure, not a scoped pattern fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    80.33s
- Tier A attempts: N/A

## Files changed
None — Tier B, no fix attempted.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
