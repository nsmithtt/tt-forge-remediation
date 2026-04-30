# Remediation Summary: firworks_qwen3_vl-pytorch-32b_thinking_nvfp4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[firworks_qwen3_vl/pytorch-32b_thinking_nvfp4-single_device-inference]

## Result
FAIL — After fixing the missing compressed-tensors requirement, the test fails with INTERNAL: Error code: 13 at grid_thw.tolist() on a TT tensor in the visual encoder. This is a Tier B device-to-host transfer bug in the TT PJRT runtime.

## Stack layer
loader, tt-xla

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
E   ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.

(After loader fix) E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Two issues:

1. **Loader bug (fixed)**: `firworks_qwen3_vl/pytorch/requirements.txt` did not exist. The model `Fireworks/Qwen3-VL-32B-Thinking-nvfp4` uses NVFP4 quantization in compressed-tensors format, which requires the `compressed-tensors` package to instantiate the quantization config during `from_pretrained`. This caused the original `ImportError`.

2. **Tier B compiler bug (unfixed)**: After installing `compressed-tensors`, the test progresses through model loading and Dynamo compilation, then fails at execution when `Qwen3VisionTransformerPretrainedModel.fast_pos_embed_interpolate` calls `grid_thw.tolist()` on a TT device tensor. The TT PJRT runtime does not support synchronous device-to-host tensor reads (error: `INTERNAL: Error code: 13`). This is not an OOM — the traceback uniquely identifies `grid_thw.tolist()` at `modeling_qwen3_vl.py:699` via `tt_torch/torch_overrides.py:34`.

The call chain is:
  `Qwen3VLForConditionalGeneration.forward` → `get_image_features` → `visual.forward` → `fast_pos_embed_interpolate` → `grid_thw.tolist()`

`image_grid_thw` is an integer LongTensor (grid dimensions for the visual encoder) that the test framework moves to TT device as part of the model inputs. When `.tolist()` is called to unpack it into Python integers, the TT runtime fails to transfer the tensor data from device to host.

## Fix
1. **tt_forge_models** (`remediation/firworks_qwen3_vl-pytorch-32b_thinking_nvfp4-single_device-inference`):
   - `firworks_qwen3_vl/pytorch/requirements.txt` — created with `compressed-tensors` dependency.

2. The device-to-host transfer bug requires implementing PJRT host-read paths for integer tensors in the TT runtime. No fix attempted.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

The TT PJRT runtime does not implement synchronous device-to-host tensor reads. Supporting `tensor.tolist()` (and the underlying `ToLiteral` PJRT call) requires new transfer infrastructure in `tt-metal`/`tt-xla`. The fix cannot be scoped to one or two files — it requires implementing the host-copy path at the PJRT buffer level.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    683.20s
- Tier A attempts: N/A

## Files changed
- `firworks_qwen3_vl/pytorch/requirements.txt` (created in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6b8e18b3f80e5ea3d0c61c94fa0a6614025a6a2a |
| tt-forge-models | d59c3af2071d9c095bfedad2af2a38bc3ad1602c |
