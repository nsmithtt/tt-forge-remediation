# Remediation Summary: autoglm_phone-conditional_generation-pytorch-autoglm_phone_9b_multilingual-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[autoglm_phone/conditional_generation/pytorch-autoglm_phone_9b_multilingual-single_device-inference]

## Result
FAIL — `torch.arange(h)` where `h` is a TT XLA device tensor fails with `INTERNAL: Error code: 13`; requires PJRT device-to-host integer transfer not supported by TT backend

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
at `transformers/models/glm4v/modeling_glm4v.py:734: in rot_pos_emb`:
```python
hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
```
where `h` is a 0-dim `torch.int64` tensor on the TT XLA device.

The original reported failure message was a `UserWarning` logged via `logger.warning_once()` in `transformers/models/auto/image_processing_auto.py:558`:
```
The image processor of type `Glm4vImageProcessor` is now loaded as a fast processor
by default, even if the model checkpoint was saved with a slow processor. This is a
breaking change and may produce slightly different outputs. To continue using the slow
processor, instantiate this class with `use_fast=False`.
```
This warning is printed to the test log but is not the test exception; the actual failure is the compiler error above.

## Root cause
`Glm4vVisionModel.rot_pos_emb` iterates over `grid_thw` (a `[N, 3]` integer tensor holding the temporal/height/width grid dimensions) with `for t, h, w in grid_thw:`, extracting 0-dim int64 device tensors. It then calls `torch.arange(h)`, where `h` is a TT XLA device tensor. Evaluating `torch.arange(end)` with a device-side `end` requires materialising the integer value on the host (a PJRT device-to-host transfer). The TT PJRT backend does not support this transfer path for integer scalars inside a compiled XLA graph, producing `INTERNAL: Error code: 13`.

The `use_fast=False` loader fix (committed to `tt_forge_models`) suppresses the unrelated image-processor warning, but it does not affect the runtime path that hits `rot_pos_emb`.

Any loader-level workaround that tries to call `.cpu()` on `grid_thw` inside the model forward also fails for the same reason: `.cpu()` on a TT tensor inside `torch.compile("tt")` forces an XLA sync that fails with the same INTERNAL error. Using `@torch.compiler.disable` to exempt `rot_pos_emb` from compilation would constitute a forbidden CPU-offload workaround for RoPE computation.

## Fix
Proposed fix lives in the `tt-xla` PJRT layer. The TT PJRT backend needs to support device-to-host transfer of scalar integer tensors during compiled graph execution. Concretely, `torch.arange(end_tensor)` where `end_tensor` is a TT device scalar should be supported by extracting the integer value via PJRT buffer access and then creating the sequence on the host (or by implementing a dynamic-shape `arange` in the XLA lowering).

The loader fix for `use_fast=False` has been committed to `tt_forge_models` on branch `remediation/autoglm_phone-conditional_generation-pytorch-autoglm_phone_9b_multilingual-single_device-inference` (commit `275eeb4e20`).

## Tier B justification
**Indicator**: new-infrastructure — implementing device-to-host transfer of integer scalars in the TT PJRT backend requires adding a new transfer path in `tt-xla`, specifically enabling `torch.arange(device_tensor)` through `aten::arange.start_step` lowering with a host-side scalar extraction, or adding a PJRT int-scalar read to the dynamo bridge.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    264.47s (first run), 211.68s (second run, after device reset)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/autoglm_phone/conditional_generation/pytorch/loader.py` — added `use_fast=False` to `AutoProcessor.from_pretrained()` to suppress the Glm4vImageProcessor fast-processor breaking change warning

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 275eeb4e20f50b3e7a194e752d49dcdf4d74ed28 |
