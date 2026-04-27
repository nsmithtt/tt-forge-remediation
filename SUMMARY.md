# Remediation Summary: asid_captioner-pytorch-3B-single_device-inference

## Skill version
10

## Test
tests/runner/test_models.py::test_all_models_torch[asid_captioner/pytorch-3B-single_device-inference]

## Result
FAIL — `grid_thw.tolist()` on TT device tensor fails with INTERNAL Error code 13 in the Qwen2.5-Omni visual encoder

## Failure
Original failure:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

After loader fix, new failure on TT device run:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  File "transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py", line 1153, in rot_pos_emb
    for t, h, w in grid_thw.tolist():
  File "python_package/tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

## Root cause

Two separate issues were found and fixed:

**Issue 1 (loader bug — fixed):** The original error was a transformers 5.x breaking change.
`AutoProcessor.from_pretrained` loaded `Qwen2VLImageProcessor` as a fast processor instead of
slow. Fix: replaced `AutoProcessor` with `Qwen2_5OmniProcessor` (which doesn't have this issue).

**Issue 2 (loader bug — fixed):** The model `AudioVisual-Caption/ASID-Captioner-3B` has
`model_type: qwen2_5_omni` in its config, but the loader was using
`Qwen2VLForConditionalGeneration`. This caused a feature-dimension mismatch between the
Qwen2.5-Omni visual encoder and the Qwen2VL language model forward pass:
```
ValueError: Image features and image tokens do not match, tokens: 551, features: 551
```
(The message shows equal counts, but the internal numel check fails because hidden_size differs.)
Fix: switched to `Qwen2_5OmniThinkerForConditionalGeneration` + `Qwen2_5OmniProcessor`.

**Issue 3 (compiler-stack bug — not fixed):** After the loader fixes, the test fails during the
TT device run inside `Qwen2_5OmniVisionEncoder.rot_pos_emb()`:

```python
for t, h, w in grid_thw.tolist():   # grid_thw is now a TT device tensor
```

The TT XLA runner moves ALL model inputs to the TT device via `tree_map(to_device)` before
execution. `image_grid_thw` (a LongTensor of image grid dimensions) is moved to the TT device.
When `rot_pos_emb` calls `grid_thw.tolist()` inside the compiled model execution, it attempts
to materialize the integer tensor from the TT device back to CPU. This fails with
"Bad StatusOr access: INTERNAL: Error code: 13".

The same pattern also appears in `get_window_index` (line 1185) and would fail there if
`rot_pos_emb` were somehow bypassed.

Layer: **compiler frontend (tt-xla)** — the TT XLA backend does not support materializing
integer device tensors to Python scalars (`.tolist()`) during compiled graph execution.

## Fix

Loader fixes applied in `tt-xla/third_party/tt_forge_models`, branch
`remediation/asid_captioner-pytorch-3B-single_device-inference`:

1. `asid_captioner/pytorch/loader.py`: replaced `AutoProcessor` with `Qwen2_5OmniProcessor`
   (not a forbidden workaround — this is the correct processor class for the model's
   `qwen2_5_omni` model_type, matching the pattern used by `qwen_2_5_omni/pytorch/loader.py`).

2. `asid_captioner/pytorch/loader.py`: replaced `Qwen2VLForConditionalGeneration` with
   `Qwen2_5OmniThinkerForConditionalGeneration` and added `model.config.use_cache = False`
   (not a forbidden workaround — fixes wrong model class for the actual checkpoint architecture).

**Proposed compiler fix:** The TT XLA backend should support `.tolist()` on integer device
tensors by transparently syncing the tensor to host before returning Python values. This would
allow the existing Python-level control flow in Qwen2.5-Omni's visual encoder to work.
Alternatively, the device runner could detect integer-typed model inputs used only in
Python-level control flow and keep them on CPU.

## Verification
pytest exit status: FAILED
Wall-clock duration: ~144 seconds (model already cached from prior runs)
Hardware: n150 (wormhole)

CPU golden run: PASSES (after loader fixes)
TT device run: FAILS at `grid_thw.tolist()` in `Qwen2_5OmniVisionEncoder.rot_pos_emb`

## Files changed
- `asid_captioner/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 14040fe173b9c1eeeae6ff7a15d73176ec5e3171 |
| tt-forge-models | 322bd4ab26becbdcdcfb62c53876b00b3085794c |
