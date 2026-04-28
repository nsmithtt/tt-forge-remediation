# Remediation Summary: glm_4_1v-conditional_generation-pytorch-glm_4_1v_9b_thinking-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_1v/conditional_generation/pytorch-glm_4_1v_9b_thinking-single_device-inference]

## Result
FAIL — INTERNAL: Error code: 13 when compiling/running `torch.arange(h)` where `h` is derived from `image_grid_thw` in the vision encoder's `rot_pos_emb`; loader `use_fast=False` fix applied but compiler-stack bug remains

## Stack layer
tt-xla

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
pjrt-device-to-host-transfer

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
The image processor of type `Glm4vImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.

After adding `use_fast=False`, the model continues to fail with:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
at `modeling_glm4v.py:734: in rot_pos_emb` → `hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)`,
intercepted by `tt_torch/torch_overrides.py:34` (`TorchFunctionMode.__torch_function__`).

## Root cause
Two issues are present:

**1. Loader (fixed):** The `Glm4vImageProcessor` is a transformers 5.x image processor that now
defaults to the fast C++ implementation. The loader's `AutoProcessor.from_pretrained()` call did
not pass `use_fast=False`, so the fast processor was loaded with a deprecation-as-breaking-change
warning. Fixed by adding `use_fast=False`.

**2. Compiler-stack bug (unfixed):** `Glm4vVisionTransformer.rot_pos_emb` iterates over the
`grid_thw` tensor with `for t, h, w in grid_thw:` and calls `torch.arange(h)` where `h` is a
0-dim integer tensor. This call is intercepted by the active `TorchFunctionMode` in
`tt_torch/torch_overrides.py` and dispatched to the TT XLA backend. The backend returns
`INTERNAL: Error code: 13`, indicating it cannot handle `torch.arange` with a device tensor as
the shape argument. This is the `pjrt-device-to-host-transfer` class of failures: extracting a
scalar integer value from a device tensor to use as a Python-level shape.

The for-loop over `grid_thw` is data-dependent control flow that causes a dynamo graph break;
after the break, `grid_thw` items are live on the TT device but `torch.arange` requires a
CPU/Python scalar as its range argument. The PJRT bridge does not implement the device→scalar
transfer path needed to satisfy this requirement.

## Fix
**Loader fix applied** in `tt-forge-models`:
- `glm_4_1v/conditional_generation/pytorch/loader.py`: added `use_fast=False` to
  `AutoProcessor.from_pretrained()` to suppress the transformers 5.x breaking change.

**Proposed compiler-stack fix (Tier B, not attempted):** Implement the device-to-scalar integer
transfer path in the TT PJRT bridge (likely in the `pjrt_implementation/` C++ layer in tt-xla),
so that `torch.arange(device_tensor)` can extract the scalar end value and proceed. This is
analogous to the existing device→host tensor transfer path but for 0-dim integer scalars used as
shape arguments.

## Tier B justification
`new-infrastructure` — the fix requires implementing a device-to-scalar transfer path in the
PJRT bridge. There is no existing mechanism to extract a scalar integer from a TT device tensor
for use as a Python-level shape argument to `torch.arange`. This is new PJRT infrastructure, not
a pattern fix in an existing lowering function.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    196.67s (0:03:16)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `glm_4_1v/conditional_generation/pytorch/loader.py` — added `use_fast=False`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c2abd6c105b0b97fe8918ad71b8d3349317ab040 |
| tt-forge-models | 09146ff80c7cb18ccd948d4f0d24e0351f399773 |
