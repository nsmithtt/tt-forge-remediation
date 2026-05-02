# Remediation Summary: medmo-image_to_text-pytorch-4b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medmo/image_to_text/pytorch-4b-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: grid_thw.tolist() on TT tensor raises INTERNAL Error code: 13; Tier B new-infrastructure

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

Full call chain:
  transformers/models/qwen3_vl/modeling_qwen3_vl.py:699 in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
  tt_torch/torch_overrides.py:34 in __torch_function__
    return func(*args, **(kwargs or {}))
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
`MBZUAI/MedMO-4B` is a Qwen3VL-based vision-language model. The test
framework moves all input tensors (including `image_grid_thw`, a LongTensor
of image grid dimensions) to the TT device. During the forward pass,
`Qwen3VLVisionTransformer.forward` calls `self.fast_pos_embed_interpolate(grid_thw)`
which calls `grid_thw.tolist()` at line 699 of `modeling_qwen3_vl.py`. The
TT PJRT runtime does not implement synchronous device-to-host tensor reads:
calling `.tolist()` triggers an internal buffer read that fails with
INTERNAL: Error code: 13.

The same call site in `rot_pos_emb` (line 660) and in `Qwen3VLModel.get_rope_index`
and `get_image_features` also call `.tolist()` on `grid_thw` / `input_ids`.

The bug is not OOM. Error code: 13 is a PJRT buffer read failure, not a
device memory exhaustion.

The only known workarounds — `@torch.compiler.disable` on
`fast_pos_embed_interpolate` or moving `grid_thw` to `.cpu()` inside
the traced methods — are both forbidden (CPU offload) or ineffective
(`.cpu()` inside Dynamo-traced methods creates a D2H op in the XLA graph
that still fails at materialization with the same Error code: 13).

## Fix
No fix applied. The correct fix is to implement synchronous device-to-host
tensor reads in the TT PJRT buffer layer (in tt-xla or tt-metal), so that
`.tolist()`, `.item()`, and similar Python-level value extraction operations
work on TT tensors.

The fix would live in the PJRT buffer implementation:
- `tt-xla/pjrt_implementation/src/api/` — specifically the buffer read path
  that handles host-readable tensor data; the TT backend needs to support
  `PjRtBuffer::ToLiteralSync()` or equivalent.

## Tier B justification
Which indicator: new-infrastructure

The TT PJRT runtime currently has no mechanism to perform synchronous
device-to-host buffer reads. Supporting `.tolist()` / `.item()` on TT tensors
requires implementing `PjRtBuffer::ToLiteralSync()` (or equivalent eager
read path) in the PJRT plugin — this is new infrastructure, not a
contained one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    80.40s
- Tier A attempts: N/A

## Files changed
None (Tier B — no fix attempted)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
