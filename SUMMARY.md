# Remediation Summary: iamjb_chexpert_mimic_cxr_impression_baseline-image_captioning-pytorch-impression-baseline-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[iamjb_chexpert_mimic_cxr_impression_baseline/image_captioning/pytorch-impression-baseline-single_device-inference]

## Result
FAIL — after loader fix, the test hangs indefinitely in the PJRT device-to-host transfer phase (pjrt-device-to-host-transfer, Tier B)

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
Test exceeded configured timeout and was killed

Reproduction: after all device program graphs compile (~120s wall clock), the process hangs indefinitely waiting for the PJRT copy-to-host operation. The test was killed by `timeout 120`. The last visible progress before the hang was the model weight loading completing and graphs compiling successfully.

## Root cause

Two distinct bugs were identified:

**1. Loader bug (fixed in this report)**

`load_inputs()` returned only `{"pixel_values": ...}` without `decoder_input_ids`. In transformers 5.2.0, `VisionEncoderDecoderModel.forward()` no longer auto-derives `decoder_input_ids` when `labels=None`; it passes `None` directly to the decoder, which raises:
```
ValueError: You must specify exactly one of input_ids or inputs_embeds
```
Fix: add `decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]])` (CLS token id = 0 for this model, matching `config.decoder_start_token_id = 0`) to `load_inputs()`. Also initializes `self.tokenizer` in `load_inputs()` if not already set.

**2. PJRT device-to-host transfer hang (Tier B — unfixed)**

After the loader fix resolves the ValueError, the test compiles all TT program graphs (~120s). The process then hangs indefinitely. The hang occurs after compilation and device execution complete, in the buffer-retrieval path — consistent with `BufferInstance::copyToHost()` in `tt-xla/pjrt_implementation/src/api/buffer_instance.cc`. This is the same `pjrt-device-to-host-transfer` fingerprint seen on both n150 (Wormhole) and p150b (Blackhole) hardware.

## Fix
**Loader fix (committed):** Added `decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]])` to `load_inputs()` in `iamjb_chexpert_mimic_cxr_impression_baseline/image_captioning/pytorch/loader.py`. Also initializes `self.tokenizer` in `load_inputs()` when not already set.

**PJRT hang (proposed, not implemented):** The fix would live in the PJRT buffer management layer:
- `tt-xla/pjrt_implementation/src/api/buffer_instance.cc` — investigate and resolve the `BufferInstance::copyToHost()` hang for VisionEncoderDecoder output on both Wormhole and Blackhole hardware
- `tt-mlir/runtime/lib/ttnn/runtime.cpp` — check whether `::ttnn::from_device` can also hang under the static-mutex pattern

## Tier B justification
**cross-cutting**: The PJRT copy-to-host hang requires coordinating changes across the PJRT buffer management layer (`tt-xla`), the runtime transfer path (`tt-mlir`), and possibly the TTNN kernel layer (`tt-metal`). The hang reproduces on both Wormhole (n150) and Blackhole (p150b) hardware, and the exact mechanism has not been fully diagnosed. The fix is unknown and diagnosis must precede implementation.

## Verification
- pytest exit: TIMEOUT (hung indefinitely, killed after 120s)
- Hardware:    blackhole-p150b
- Duration:    ~120s compilation + indefinite hang
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/iamjb_chexpert_mimic_cxr_impression_baseline/image_captioning/pytorch/loader.py` (loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7cdf4338e660bd10cf353e74d3c59001eed05fcc |
| tt-forge-models | 4d4d7d60320a954d8c86664caa654f6babc7fcd7 |
