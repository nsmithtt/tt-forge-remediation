# Remediation Summary: llava_onevision-pytorch-Qwen2_7B_SI-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision/pytorch-Qwen2_7B_SI-single_device-inference]

## Result
FAIL — Tier B compiler-stack bug: INTERNAL error 13 during XLA graph compilation triggered by `get_placeholder_mask` dynamic boolean indexing (SigLIP patch embedding in accumulated graph)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
xla-internal-error-13-siglip-patch-embed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (2026-04-21, branch arch-c-36-tt-xla-dev/nsmith/hf-bringup-2):

    2026-04-21 13:59:37.410 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

Reproduced failure (2026-05-01, after applying use_fast=False loader fix, 207.14s):

    RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

The TIMEOUT in the original run and INTERNAL error 13 in the reproduction are different manifestations of the same underlying failure: XLA graph compilation/execution hangs or fails when the accumulated graph (including SigLIP patch embedding) is synced to device.

## Root cause
Two separate bugs, addressed in order:

1. **Loader bug (fixed)**: transformers 5.x changed the default for `AutoProcessor.from_pretrained`
   to load fast image processors. The LLaVA-OneVision loader was calling `from_pretrained` without
   `use_fast=False`, producing a warning about breaking-change behaviour. Fixed by adding
   `use_fast=False` to `_load_processor()`.

2. **Compiler-stack bug (unfixed, Tier B)**: After the loader fix, silicon compilation of the
   LLaVA-OneVision forward graph fails with `INTERNAL: Error code: 13`. The call site is
   `get_placeholder_mask` (modeling_llava_onevision.py:460–461) where `torch_compilable_check`
   triggers `torch.compile` to compile and sync the buffered XLA graph:

       torch_compilable_check(
           inputs_embeds[special_image_mask].numel() == image_features.numel(), ...
       )

   `inputs_embeds[special_image_mask]` is a dynamic boolean-indexed gather with a
   data-dependent output shape (number of image tokens × hidden_size). The accumulated XLA
   graph at the sync point includes the SigLIP vision tower (Conv2D patch embedding:
   kernel=14×14, stride=14×14, input=[5,3,384,384], output_channels=1152). Silicon
   compilation and execution of this graph fails with INTERNAL:13.

   This is the same root cause as the 0.5B_SI variant (see
   `report/llava-onevision-pytorch-Qwen2_0.5B_SI-single_device-inference`), which also reached
   INTERNAL:13 at the same `get_placeholder_mask` sync point after the use_fast fix.

## Fix
Loader fix applied:
- `tt-xla/third_party/tt_forge_models/llava_onevision/pytorch/loader.py`: added `use_fast=False`
  to `AutoProcessor.from_pretrained()` in `_load_processor()`.

Committed as `26120f0e9c` on branch
`remediation/llava_onevision-pytorch-Qwen2_7B_SI-single_device-inference` in tt-forge-models.

No compiler-stack fix was attempted. The INTERNAL:13 error on silicon at the SigLIP patch
embedding requires further diagnosis before a Tier A fix can be scoped.

## Tier B justification
internal-error-unknown-mechanism

The `INTERNAL: Error code: 13` from silicon is the same undiagnosed error as in the 0.5B_SI
report. The XLA graph sync at `get_placeholder_mask` fails during program creation for the
accumulated graph that includes the SigLIP Conv2D patch embedding. The actual mechanism
(OOM, CB overflow, or other hardware error) is unknown; diagnosis must precede any fix
attempt.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    207.14s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llava_onevision/pytorch/loader.py` — added `use_fast=False`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b6f983b2294fbd6ffb596ae3fa86f0fc73363873 |
| tt-forge-models | 26120f0e9c08f114a6c2424d4f69b625a5e8515e |
