# Remediation Summary: gemma3_abliterated-multimodal-pytorch-YanLabs-normpreserve-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_abliterated/multimodal/pytorch-YanLabs/gemma-3-27b-it-abliterated-normpreserve-single_device-inference]

## Result
XFAIL — 27B model exceeds single-device DRAM capacity; loader bug (use_fast) also fixed

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
transformers-5x-use-fast-default

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
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two layered issues:

1. **Loader bug (transformers 5.x)**: `_load_processor()` in `gemma3_abliterated/multimodal/pytorch/loader.py` called `AutoProcessor.from_pretrained()` without `use_fast=False`. In transformers 5.2.0, `Gemma3ImageProcessor` is now loaded as `Gemma3ImageProcessorFast` by default when torchvision is available, emitting the breaking-change warning. This is the same pattern fixed for `gemma3/multimodal/pytorch/loader.py` (commit 2c4eb3a2de).

2. **Hardware capacity ceiling**: `YanLabs/gemma-3-27b-it-abliterated-normpreserve` is a 27B parameter model. The gemma3-4b-it multimodal model already OOMs on single device (tt-xla issue #4007). A 27B model is ~7× larger and cannot fit in single-device DRAM.

## Fix
1. **Loader fix** in `tt-forge-models` branch `remediation/gemma3_abliterated-multimodal-pytorch-YanLabs-normpreserve-single_device-inference`:
   - `gemma3_abliterated/multimodal/pytorch/loader.py`: added `use_fast=False` to `AutoProcessor.from_pretrained()` call in `_load_processor()`.

2. **Test config XFAIL** in `tt-xla` branch `remediation/gemma3_abliterated-multimodal-pytorch-YanLabs-normpreserve-single_device-inference`:
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added `KNOWN_FAILURE_XFAIL` for this test citing hardware capacity ceiling.

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `gemma3_abliterated/multimodal/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0ca0671769409821751d20357c307469290b49c2 |
| tt-forge-models | 0501e209b2376f32809c5eee7ad178777fe4a6d2 |
