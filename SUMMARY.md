# Remediation Summary: emu3_gen-pytorch-Gen-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[emu3/gen/pytorch-Gen-single_device-inference]

## Result
SILICON_PASS — all loader-layer transformers 5.x bugs fixed; test passes on TT silicon in 120.56s

## Stack layer
loader

  - `loader` — bug was in tt_forge_models or test inputs

## Tier
N/A

## Bug fingerprint
transformers-5x-rope-meta-device-uninit-buffers

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: "The image processor of type `Emu3VisionVQImageProcessor` is now loaded as a fast
processor by default, even if the model checkpoint was saved with a slow processor. This is a
breaking change..."

After early fixes, subsequent failure: "PCC comparison failed. Calculated: pcc=nan (invalid
value). Required: pcc=0.99."

## Root cause
Seven sequential transformers 5.x breaking changes in the Emu3-Gen loader, the most critical
being: transformers 5.x uses `init_empty_weights()` (meta device) during `from_pretrained`.
`persistent=False` buffers (`inv_freq`, `cos_cached`, `sin_cached`) are not stored in the
checkpoint, so they come out of `from_pretrained` as uninitialized float32 tensors.  When
the forward pass runs, the RoPE embedding returns NaN `cos`/`sin` values, propagating NaN
through all 32 attention layers and producing NaN logits on CPU (making PCC undefined), while
TT silicon was correctly handling the computation because the kimi_k2 loader's
`is_torch_fx_available` shim was already applied by the time Emu3's remote module code ran.

The full list of loader bugs fixed (all in `tt_forge_models/emu3/gen/pytorch/loader.py`):

1. **`use_fast=False`**: `AutoImageProcessor.from_pretrained` now defaults to fast processor.
2. **`ProcessorMixin.get_attributes()` signature inspection**: transformers 5.x inspects
   `__init__` signatures — `vision_tokenizer` is detected as a third modality attribute but
   `Emu3Processor` only passes 2 args to `super().__init__()`. Fix: override
   `get_attributes` classmethod to return the 2 actual attributes.
3. **`tokenizer.encode(list)` returns nested lists**: transformers 5.x returns `[[id], ...]`
   instead of `[id, ...]`. `build_const_helper` expects flat list. Fix: wrap `encode` to
   flatten one level.
4. **`config.rope_scaling` key `"type"` vs `"rope_type"`**: transformers 5.x normalizes
   `rope_theta` into `rope_scaling = {'rope_theta': ..., 'rope_type': 'default'}` but remote
   `modeling_emu3.py` reads `rope_scaling["type"]`. Fix: set `config.rope_scaling = None`
   before loading.
5. **`image_size` in processor output**: processor always returns `image_size` in generation
   mode but `forward()` doesn't accept it. Fix: `result.pop("image_size", None)`.
6. **`DynamicCache.get_usable_length()` removed**: transformers 5.x removed this method.
   Fix: `result["use_cache"] = False` for single forward-pass inference.
7. **RoPE cache uninitialized (NaN root cause)**: `init_empty_weights()` leaves
   `persistent=False` buffers (`inv_freq`, `cos_cached`, `sin_cached`) uninitialized.
   Fix: `_reinit_rope_caches()` recomputes them in the correct dtype after loading.
8. **`is_torch_fx_available` removed**: transformers 5.x removed this symbol from
   `transformers.utils.import_utils`. Remote `modeling_emu3.py` imports it at module level.
   Fix: inject a shim `(returns False)` at loader import time, mirroring the kimi_k2 pattern.

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch
`remediation/emu3_gen-pytorch-Gen-single_device-inference`:

- `emu3/gen/pytorch/loader.py`: all eight fixes above
- `emu3/gen/pytorch/requirements.txt`: added `tiktoken` (missing dependency)

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    120.56s (0:02:00)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/emu3/gen/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/emu3/gen/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bbe43dcb6675342370d8daaf472169c34eb5fb2c |
| tt-forge-models | 00250766c8fd91f6bcf8af7e9d78c3de3a5c27f9 |
