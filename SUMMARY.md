# Remediation Summary: chexpert_mimic_cxr_impression_baseline-image_to_text-pytorch-Baseline-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chexpert_mimic_cxr_impression_baseline/image_to_text/pytorch-Baseline-single_device-inference]

## Result
FAIL — PJRT device-to-host transfer hangs indefinitely in `::ttnn::from_device(tensor, blocking=true)` on Blackhole after model execution completes

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
Test exceeded configured timeout and was killed

Reproduction: after all 463 device programs execute (~67s wall clock), the process hangs indefinitely. Stack trace shows 249 threads in `futex_wait_queue`: one thread holds `s_copy_to_host_internal_mutex` (static global) inside `move_to_host()`, while 248 threads wait to acquire the same mutex. The holding thread never returns from `::ttnn::from_device(inputTensor, /*blocking=*/true)`.

## Root cause

The failure has two components:

**1. Loader issue (pre-existing, already fixed in CI branch)**

On the base branch commit `0f7b734348`, `load_inputs()` returned only `{"pixel_values": ...}` without `decoder_input_ids`. With transformers 5.2.0, `VisionEncoderDecoderModel.forward()` no longer auto-derives `decoder_input_ids`; it passes `None` directly to `BertGenerationDecoder`, which strictly validates `input_ids is not None` and raises `ValueError`. The CI branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-39` already fixes this by adding `decoder_input_ids = torch.tensor([[self.tokenizer.cls_token_id]])` to `load_inputs()`.

**2. PJRT device-to-host transfer hang (the actual CI failure)**

After all computation completes on silicon, result buffers are transferred to host via `BufferInstance::copyToHost()` in `tt-xla/pjrt_implementation/src/api/buffer_instance.cc`. Each buffer spawns a thread that:
1. Acquires `s_copy_to_host_internal_mutex` (a **static global mutex** serializing all buffer transfers)
2. Calls `m_pjrt_tensor->move_to_host()` → `tt::runtime::toHost(tensor, untilize=true)` → `toHostSingleTensor()` in `tt-mlir/runtime/lib/ttnn/runtime.cpp`

In `toHostSingleTensor()`, the Blackhole workaround disables on-device untilization:
```cpp
if (::tt::runtime::workaround::Env::get().blackholeWorkarounds) {
    untilizeOnDevice &= getArch() != ::tt::target::Arch::Blackhole;
}
```
This forces `untilize=true, untilizeOnDevice=false`, which then forces `blocking=true`:
```cpp
if (untilize && !blocking) {
    LOG_WARNING("Overriding blocking parameter to true...");
    blocking = true;
}
```
The resulting call `::ttnn::from_device(inputTensor, /*blocking=*/true)` hangs indefinitely on Blackhole. Because `s_copy_to_host_internal_mutex` is static, all 248 remaining transfer threads stall waiting for the deadlocked thread, and the process never terminates.

The VisionEncoderDecoder model produces 249 output buffers (logits + encoder hidden states + decoder key/value caches). One buffer's transfer deadlocks and the entire result read-back hangs.

## Fix
No fix attempted. Proposed fix would need to be in the PJRT/TTNN transfer path:
- `tt-xla/pjrt_implementation/src/api/buffer_instance.cc` — either remove/replace `s_copy_to_host_internal_mutex`, or make `move_to_host()` non-blocking with proper async signaling
- `tt-mlir/runtime/lib/ttnn/runtime.cpp` — investigate why `::ttnn::from_device(tensor, blocking=true)` hangs on Blackhole; the Blackhole workaround may be masking a deeper issue with on-device untilization that needs a different solution
- Cross-cutting: the static mutex pattern couples all buffer transfers; removing it requires understanding whether `::ttnn::from_device` is thread-safe on Blackhole

## Tier B justification
**cross-cutting**: The fix requires coordinating changes across the PJRT buffer management layer (`tt-xla/pjrt_implementation/src/api/buffer_instance.cc`), the runtime transfer path (`tt-mlir/runtime/lib/ttnn/runtime.cpp`), and possibly the TTNN kernel layer in `tt-metal`. The static global mutex exists to serialize calls that may not be thread-safe; removing it without understanding whether `::ttnn::from_device` is reentrant on Blackhole risks data corruption. Additionally, the root cause of *why* `from_device(blocking=true)` deadlocks on Blackhole is not yet established — the fix is unknown and diagnosis must precede implementation.

## Verification
- pytest exit: TIMEOUT
- Hardware:    blackhole-p150b
- Duration:    ~67s execution + indefinite hang (killed after >35 minutes)
- Tier A attempts: N/A

## Files changed
None (Tier B FAIL — no fix attempted)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 986e808f120435cc52565a0d853562e522d7c924 |
