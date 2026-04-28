# Remediation Summary: llava-onevision-pytorch-Qwen2_0.5B_SI-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision/pytorch-Qwen2_0.5B_SI-single_device-inference]

## Result
FAIL — Tier B compiler-stack bug: INTERNAL error 13 during XLA compilation of LLaVA-OneVision forward pass (includes SigLIP Conv2D patch embedding)

## Stack layer
tt-metal

  - `loader`         — bug was in tt_forge_models or test inputs
  - `tt-xla`         — bug in compiler frontend (PJRT, torch_xla bridge)
  - `tt-mlir`        — bug in compiler core (StableHLO→TTIR lowering)
  - `tt-metal`       — bug in backend runtime / kernels
  - `hardware-class` — model exceeds single-device capacity (XFAIL)
  - `n/a`            — NO_FIX_NEEDED (could not reproduce)

## Tier
B

  - `N/A` — loader fix, no fix needed, or hardware-class XFAIL
  - `A`   — compiler-stack fix attempted (succeeded → SILICON_PASS,
            ran out of attempts → FAIL with explanation)
  - `B`   — compiler-stack bug filed without attempting fix

## Bug fingerprint
xla-internal-error-13-siglip-patch-embed

  Format: `<area>-<short-description>`. Use the same string verbatim
  whenever a later report hits the same bug — this is how the audit
  groups failures.

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
The original reported failure was a transformers 5.x warning raised as an error in the test harness:

    The image processor of type `LlavaOnevisionImageProcessor` is now loaded as a fast
    processor by default, even if the model checkpoint was saved with a slow processor.
    This is a breaking change and may produce slightly different outputs. To continue
    using the slow processor, instantiate this class with `use_fast=False`.

After applying the loader fix (use_fast=False), the test runs 96.55 s and fails with:

    INTERNAL: Error code: 13

The error is triggered inside `get_placeholder_mask` (modeling_llava_onevision.py:460) at a
`torch_compilable_check` call, which causes torch.compile to compile the buffered XLA graph.
That graph includes the SigLIP vision tower with its Conv2D patch embedding
(kernel=14×14, stride=14×14, input=[5,3,384,384], output_channels=1152). Silicon
compilation of that graph fails with INTERNAL error code 13.

## Root cause
Two separate bugs, addressed in order:

1. **Loader bug (fixed)**: transformers 5.x changed the default for `AutoProcessor.from_pretrained`
   to load fast image processors. The LLaVA-OneVision loader in
   `llava_onevision/pytorch/loader.py` was calling `from_pretrained` without `use_fast=False`,
   producing a breaking-change warning/error. Fixed by adding `use_fast=False` to the
   `_load_processor()` call.

2. **Compiler-stack bug (unfixed, Tier B)**: After the loader fix, silicon compilation of the
   LLaVA-OneVision forward graph fails with `INTERNAL: Error code: 13`. The graph includes
   the SigLIP patch embedding Conv2D:
   - kernel = 14×14, stride = 14×14
   - input = [5, 3, 384, 384] (5 images of 384×384 after pixel_values reshape)
   - output_channels = 1152
   - `384 % 14 = 6 ≠ 0` → `auto_enable_kernel_folding` returns false → HEIGHT_SHARDED path

   The exact mechanism causing the INTERNAL:13 error was not confirmed. An initial hypothesis
   of CB overflow from `enable_fully_buffered_weights` (which multiplies weight_block_num_tiles
   by `kernel_size[0]=14` when `num_blocks_act_h > 1`) was investigated but does not hold
   under correct arithmetic:
   - `out_nhw_ntiles = ceil(5×27×27 / 32) = 114`
   - `find_closest_largest_divisor(114, 110, 1)` → 57 cores
   - `per_core_nhw_ntiles = 2`, `act_block_h_ntiles = 2`, `num_blocks_act_h = 1`
   - `enable_fully_buffered_weights = false`

   The actual cause of the INTERNAL:13 error requires further diagnosis.

## Fix
Loader fix applied:
- `tt-xla/third_party/tt_forge_models/llava_onevision/pytorch/loader.py`: added `use_fast=False`
  to `AutoProcessor.from_pretrained()` in `_load_processor()`.

Committed as `ef4822583b` on branch
`remediation/llava-onevision-pytorch-Qwen2_0.5B_SI-single_device-inference` in tt-forge-models.

No compiler-stack fix was attempted. The INTERNAL:13 error requires diagnosis of the tt-metal
Conv2D program creation for the SigLIP patch embedding configuration before a fix can be
scoped.

## Tier B justification
internal-error-unknown-mechanism

The `INTERNAL: Error code: 13` from silicon is a TT device-level error emitted during program
creation for the SigLIP patch embedding Conv2D. The CB overflow hypothesis (via
`enable_fully_buffered_weights × kernel_size[0]=14`) was ruled out by correct arithmetic
(`num_blocks_act_h = 1`). The actual mechanism is unknown; diagnosis must precede any fix
attempt.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    96.55s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llava_onevision/pytorch/loader.py` — added `use_fast=False`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fe696250476452281d7235e66fc9ac2458d92503 |
| tt-forge-models | ef4822583b760a113f8f11f9c0c54a1ea66caf0d |
