# Remediation Summary: lingshu/pytorch-32B-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[lingshu/pytorch-32B-single_device-inference]

## Result
XFAIL — 32B model in bfloat16 (~64 GB) exceeds single-device DRAM on p150b (24 GB); hardware capacity ceiling

## Stack layer
hardware-class

  - `loader`         — bug was in tt_forge_models or test inputs
  - `tt-xla`         — bug in compiler frontend (PJRT, torch_xla bridge)
  - `tt-mlir`        — bug in compiler core (StableHLO→TTIR lowering)
  - `tt-metal`       — bug in backend runtime / kernels
  - `hardware-class` — model exceeds single-device capacity (XFAIL)
  - `n/a`            — NO_FIX_NEEDED (could not reproduce)

## Tier
N/A

  - `N/A` — loader fix, no fix needed, or hardware-class XFAIL
  - `A`   — compiler-stack fix attempted (succeeded → SILICON_PASS,
            ran out of attempts → FAIL with explanation)
  - `B`   — compiler-stack bug filed without attempting fix

## Bug fingerprint
hardware-capacity-32b-exceeds-single-device-dram

  Format: `<area>-<short-description>`. Use the same string verbatim
  whenever a later report hits the same bug — this is how the audit
  groups failures.

  Examples drawn from existing failures:
    sdpa-k-chunk-size-lt-32
    pjrt-device-to-host-transfer
    conv2d-reader-indices-cb-page-size
    stablehlo-round-nearest-even-no-lowering
    aten-slice-tensor-out-of-bounds-start
    avg-pool2d-ceil-mode-zero-output
    ttmlir-f32-precision-not-preserved
    transformers-5x-use-fast-default       (loader bug)
    gguf-load-checkpoint-model-to-load-kwarg (loader bug)
    n/a                                    (NO_FIX_NEEDED only)

## Workaround self-check
Verify each rule from the Forbidden workarounds section. Mark NO if the
fix did not use the technique; YES with a measured justification only
when explicitly permitted (currently only PCC lowering for measured bf16
accumulation).

- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Reported failure message:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```

Actual reproduced failure (first error encountered):
```
TypeError: Qwen2_5_VLForConditionalGeneration.__init__() got an unexpected keyword argument 'use_cache'
```

After loader fix, hardware-level OOM:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
(raised in `rot_pos_emb` during `grid_thw.tolist()` — first device tensor operation during vision encoder forward pass)

## Root cause
Two loader bugs were found and fixed, then the model hit a hardware capacity ceiling:

1. **Loader bug (use_cache)**: `load_model()` passed `"use_cache": False` in `model_kwargs` to `Qwen2_5_VLForConditionalGeneration.from_pretrained()`. In transformers 5.2.0, `Qwen2_5_VLForConditionalGeneration.__init__()` only accepts `config` — extra kwargs are no longer silently forwarded. Fix: removed `use_cache` from `model_kwargs` and set `model.config.use_cache = False` after loading.

2. **Loader bug (use_fast)**: `_load_processor()` called `AutoProcessor.from_pretrained()` without `use_fast=False`. In transformers 5.2.0, `Qwen2VLImageProcessor` now defaults to the fast processor, which is a breaking change for models saved with the slow processor. Fix: added `"use_fast": False` to `processor_kwargs`.

3. **Hardware capacity ceiling**: After fixing the loader, the 32B model (Lingshu-32B, a Qwen2.5-VL based model) fails with `INTERNAL: Error code: 13` (OOM) on the first device operation. Lingshu-32B in bfloat16 requires ~64 GB of DRAM. The p150b device has 24 GB. This exceeds single-device capacity with no allocator bug — it is a genuine hardware class limitation.

## Fix
**Loader fixes** (in `tt_forge_models` on branch `remediation/lingshu-pytorch-32b-single_device-inference`):
- `tt-xla/third_party/tt_forge_models/lingshu/pytorch/loader.py`:
  - Removed `"use_cache": False` from `model_kwargs` in `load_model()`
  - Added `model.config.use_cache = False` after `from_pretrained()`
  - Added `"use_fast": False` to `processor_kwargs` in `_load_processor()`

**XFAIL config** (in `tt-xla` on branch `remediation/lingshu-pytorch-32b-single_device-inference`):
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  - Added `lingshu/pytorch-32B-single_device-inference` with `status: KNOWN_FAILURE_XFAIL`

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    111.76s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/lingshu/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 69a00cceaeb88f1173a8392a518e57e24ea6231e |
| tt-forge-models | 9377bb45b9b6dcc62cc9100c887af920f190bde7 |
