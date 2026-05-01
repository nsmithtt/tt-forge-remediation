# Remediation Summary: mcdse-embedding-pytorch-2B_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mcdse/embedding/pytorch-2B_v1-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer in Qwen2VL rot_pos_emb (Tier B new-infrastructure)

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

Raised at:
  transformers/models/qwen2_vl/modeling_qwen2_vl.py:744 in rot_pos_emb
    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
  ^-- h is a scalar unpacked from image_grid_thw which is on the TT device

The CI-reported failure message ("The image processor of type `Qwen2VLImageProcessor`
is now loaded as a fast processor by default...") is the text of a transformers
UserWarning printed to stderr during processor loading.  It is NOT a Python exception
and does NOT cause the test to fail.  Attempting `use_fast=False` to silence this
warning causes a harder failure: `ValueError: size must contain 'shortest_edge' and
'longest_edge' keys.` because the checkpoint format is only valid for the fast
processor.  The warning is harmless; the actual blocking failure is the
pjrt-device-to-host-transfer error described above.

## Root cause
`Qwen2VLForConditionalGeneration` (used by MCDSE for multimodal document embedding)
passes `image_grid_thw` as an input tensor.  The test infrastructure moves all input
tensors to the TT device.  Inside `Qwen2VisionTransformerPretrainedModel.rot_pos_emb`,
the code iterates:

    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h)  # h is a TT scalar → device-to-host

Python unpacking of a TT tensor requires reading individual integer values from the
device.  The TT PJRT layer does not support synchronous scalar device-to-host
transfer, so it raises INTERNAL Error code: 13.  This is the same bug that appears
in Qwen3-VL fast_pos_embed_interpolate and other vision models that use grid shape
metadata as Python integers.

## Fix
No fix possible at the loader layer:
- `use_fast=False` in AutoProcessor.from_pretrained breaks loading with ValueError.
- Keeping `image_grid_thw` on CPU would require the test framework to support
  mixed-device inputs (it does not).

Proposed compiler-stack fix: implement synchronous scalar device-to-host transfer
in the TT PJRT layer so that `.item()` / Python-scalar unpacking from TT tensors
works.  This is the same fix tracked across multiple Tier B reports under the
fingerprint `pjrt-device-to-host-transfer`.

## Tier B justification
new-infrastructure — fixing device-to-host scalar transfer requires implementing a
new code path in the PJRT transport layer that does not currently exist.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    93.11s (1:33)
- Tier A attempts: N/A

## Files changed
tt_forge_models:
- mcdse/embedding/pytorch/loader.py — investigated use_fast=False (reverted; slow
  processor incompatible with checkpoint format)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 34811554fef1fc0e8678264e940a0922f5aa93af |
