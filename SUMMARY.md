# Remediation Summary: medgemma-multimodal-pytorch-unsloth-medgemma-4b-it-unsloth-bnb-4bit-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medgemma/multimodal/pytorch-unsloth/medgemma-4b-it-unsloth-bnb-4bit-single_device-inference]

## Result
FAIL — PCC=0.860 vs required 0.99; ttmlir-bf16-matmul-precision-floor Tier B after three loader fixes and one Tier A compiler fix

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast
processor by default, even if the model checkpoint was saved with a slow
processor. This is a breaking change and may produce slightly different
outputs. To continue using the slow processor, instantiate this class with
`use_fast=False`.

## Root cause
Four separate bugs chained. All were root-caused and three fixed:

**Bug 1 — Loader (transformers 5.x breaking change):** `AutoProcessor.from_pretrained`
now defaults to the fast image processor for `Gemma3ImageProcessor`. The
loader passed no `use_fast` kwarg, triggering a breaking-change warning
that was the CI-reported failure.

**Bug 2 — Loader (missing dependency):** The `unsloth/medgemma-4b-it-unsloth-bnb-4bit`
variant requires `bitsandbytes>=0.46.1` to load, but the loader had no
`requirements.txt`. Without the package, `from_pretrained` raises
`ImportError: bitsandbytes 4-bit quantization requires bitsandbytes`.

**Bug 3 — Loader (BNB NF4 CPU inference):** `Gemma3ForConditionalGeneration`
with `device_map="cpu"` loads 394 `Linear4bit` layers. Of these, 143
skip-module layers have full-precision bfloat16 weights and 251 have
packed NF4 weights. `Linear4bit.forward()` calls
`fix_4bit_weight_quant_state_from_module`, which asserts
`weight.shape[1] == 1`; the 143 full-precision layers fail this assert.
The fix (per the AWQ dequantize-to-Linear pattern) replaces all
`Linear4bit` instances with `nn.Linear` after loading: packed layers via
`bitsandbytes.functional.dequantize_4bit`, full-precision layers
directly. CPU dequantization of NF4 via the BNB functional API works
without CUDA.

**Bug 4 — tt-xla Tier A (slice OOB):** Gemma3 sliding-window attention
(window=1024) issues `full_value_states[:, :, -1023:, :]` in
`cache_utils.py`. On a 274-token tensor, `start=-1023 < -274`, which
PyTorch eager clamps silently but the XLA backend rejects with
"Value out of range". Fix: `clamp_out_of_range_slice_starts` FX pass
in `tt-xla/python_package/tt_torch/backend/passes.py`, wired into
`torch_pass_pipeline` in `backend.py`.

**Bug 5 — tt-mlir Tier B (BF16 precision floor):** After all four fixes,
PCC=0.860 vs required 0.99 on Blackhole p150b. Gemma3 4B has 36
transformer layers and a very wide MLP (intermediate_size=10240). The
accumulated BF16 matmul rounding error on Wormhole/Blackhole hardware
computes to a precision gap identical in character to the known
`ttmlir-bf16-matmul-precision-floor` seen on Qwen3 4B (PCC=0.864,
n150), Gemma 7B (PCC~0.915), and other large language models.

## Fix
**tt_forge_models / medgemma/multimodal/pytorch/loader.py:**
- Added `use_fast=False` to `AutoProcessor.from_pretrained()`.
- Added `_dequantize_bnb_linear4bit(model)` static method that replaces
  all `bnb.nn.Linear4bit` layers with `nn.Linear` using
  `dequantize_4bit()` for packed layers and direct weight reuse for
  full-precision layers.
- Called `_dequantize_bnb_linear4bit` in `load_model` for the BNB
  variant.

**tt_forge_models / medgemma/multimodal/pytorch/requirements.txt (new):**
- `bitsandbytes>=0.46.1`

**tt-xla / python_package/tt_torch/backend/passes.py:**
- New `clamp_out_of_range_slice_starts(gm)` FX pass: for each
  `aten.slice.Tensor` node with a static negative `start` less than
  `-dim_size`, clamps `start` to `-dim_size`. Matches PyTorch eager
  semantics.

**tt-xla / python_package/tt_torch/backend/backend.py:**
- Imported `clamp_out_of_range_slice_starts` from passes.
- Called it just before `bypass_assert_tensor_metadata` in
  `torch_pass_pipeline`.

**Proposed fix for Bug 5 (not attempted — Tier B):**
Add F32 compute path for BF16 matmul accumulation in the tt-mlir/tt-metal
matmul kernels, or use `math_fidelity=HiFi4` globally for BF16 compute.
This would require cross-cutting changes to tt-metal kernel parameters.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting
BF16 matmul precision accumulation error (PCC=0.860) affects all matmul
operations through the model; fixing it requires changing compute
fidelity or accumulation type globally in tt-metal, touching multiple
kernel files across two repos.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    312.08s (0:05:12) for the last run that produced PCC=0.860
- Tier A attempts: 1

## Files changed
- tt_forge_models/medgemma/multimodal/pytorch/loader.py
- tt_forge_models/medgemma/multimodal/pytorch/requirements.txt (new)
- tt-xla/python_package/tt_torch/backend/passes.py
- tt-xla/python_package/tt_torch/backend/backend.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 753be15f4a5872ac9dc6a357543d27aa61a4ff58 |
| tt-forge-models | f23317711f5921aa2ad01e6a735189c147134f94 |
