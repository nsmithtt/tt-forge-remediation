# Remediation Summary: molmo2-conditional_generation-pytorch-molmo2_o_7b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[molmo2/conditional_generation/pytorch-molmo2_o_7b-single_device-inference]

## Result
FAIL — TTNN bool reduction overcounts by 1 due to stale tiled-buffer padding; `is_image_patch.sum()` returns N+1 on compiled TT device path (assertion fails in `build_input_embeddings`)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttnn-bool-sum-stale-tile-padding-overcount

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: assertion error
  File ".../modeling_molmo2.py", line 1446, in build_input_embeddings
    assert is_image_patch.sum() == len(image_features)
```

On compiled execution, `is_image_patch.sum()` (a boolean tensor reduction) returns 956 while
the actual number of image patch tokens in the sequence is 955. The eager
InputCollector pass preceding compilation returns the correct value 955.

The reported CI failure message (`The image processor of type 'Molmo2ImageProcessor' is
now loaded as a fast processor by default...`) was the precursor symptom; it was fixed as
a loader bug. The terminal FAIL is the TTNN bool-sum overcount at compiled inference time.

## Root cause

**Loader bugs (fixed — transformers 5.x breaking changes):**

1. `AutoProcessor.from_pretrained` defaults to `use_fast=True` in transformers 5.x, which
   causes `Molmo2ImageProcessor` to raise a warning and may load a different processor.
   Fix: pass `use_fast=False`.

2. `Molmo2Processor.__init__` passes `image_use_col_tokens` (and similar optional
   attributes) to `ProcessorMixin.__init__()`. transformers 5.x now rejects unrecognized
   kwargs with `TypeError: Unexpected keyword argument image_use_col_tokens`.
   Fix: monkey-patch `ProcessorMixin.__init__` to pop `optional_attributes` before the
   strict validator, then restore them as instance attrs after.

3. `ROPE_INIT_FUNCTIONS["default"]` was removed from transformers 5.x.
   Fix: inject `_default_rope_init` via `ROPE_INIT_FUNCTIONS.setdefault("default", ...)`.

4. `Molmo2Config` no longer carries a top-level `use_cache` attribute, causing
   `AttributeError` in the compilation path.
   Fix: set `model.config.use_cache = False` after `from_pretrained`.

5. `build_batched_images` performs integer products (e.g., 23×33=759) on device tensors
   that are BF16 on the TT device. BF16 rounds 759 → 760 and 955 → 956, breaking the
   pooled-patch count assertion.
   Fix: monkey-patch `build_batched_images` to perform all count arithmetic in Python
   ints via `.cpu().tolist()`.

**Terminal compiler bug (Tier B):**

After the loader fixes, the model loads and runs through the eager InputCollector pass
correctly (955 image patch tokens counted correctly). On the second call — the compiled
TTNN graph — `is_image_patch.sum()` over a boolean tensor of length 2064 (not a multiple
of 32) returns 956 instead of 955.

TTNN boolean reduction tiles the input to 32×32 TTNN tiles. When the input length is not
a multiple of 32 (2064 mod 32 = 16), the last tile is padded. `tilize_with_val_padding`
short-circuits when the tensor is already in TILE layout, leaving stale uninitialized
cells in the padding region. One stale padding cell evaluates to `true`, inflating the
sum by 1.

This is the same mechanism documented in memory as "TTNN tiled-buffer padding
uninitialized" / "bool sum returns N+1 (Tier B)". The fix would require
`tilize_with_val_padding` (or the underlying reduce kernel) to zero-fill the padding
cells unconditionally before reduction. That change is in tt-metal, touches the core
tiling infrastructure, and is cross-cutting.

The loader-level `_patched_build_batched_images` fixes the earlier BF16 rounding on
the pre-model data path; it cannot fix the boolean reduction inside the compiled forward.

## Fix
Loader fixes (committed to `remediation/molmo2-conditional_generation-pytorch-molmo2_o_7b-single_device-inference`
in `tt-forge-models`):

- `molmo2/conditional_generation/pytorch/loader.py`:
  - `_load_processor`: add `use_fast=False`; monkey-patch `ProcessorMixin.__init__` to
    pop `optional_attributes` before the strict validator.
  - `load_model`: inject `_default_rope_init` into `ROPE_INIT_FUNCTIONS`; set
    `model.config.use_cache = False`; monkey-patch `model.model.build_batched_images`
    to perform all count arithmetic on CPU using Python ints.

Two commits:
- `6161cd9184` — Fix Molmo2 processor loading and device compatibility for transformers 5.x
- `53aed5b672` — Fix BF16 rounding in _patched_build_batched_images: use Python ints throughout

**Proposed fix for the Tier B bug (not implemented):**
In tt-metal, `tilize_with_val_padding` must zero-fill the padding region of the last
tile unconditionally before the reduction kernel reads it. The relevant code is in
`tt_eager/tt_dnn/op_library/tilize/tilize_with_val_padding/...` (or equivalent in the
TTNN infrastructure). The fix requires ensuring the padding cells are initialized to
the identity element of the reduction (0 for sum, `false` for boolean AND).

## Tier B justification

**Indicator:** cross-cutting

The `tilize_with_val_padding` pad-cell initialization affects every boolean (and
potentially non-boolean) reduction on non-tile-aligned inputs. Fixing it requires
changes to the tt-metal tiling infrastructure that are used across many ops, making
it a cross-cutting change beyond the scope of a single-PR loader fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    not recorded (failed mid-run)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/molmo2/conditional_generation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 53aed5b672d322e80603294338b7c94234652740 |
