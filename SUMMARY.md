# Remediation Summary: gemma3_bnb_4bit-270M_Unsloth_BNB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_bnb_4bit/pytorch-270M_Unsloth_BNB_4bit-single_device-inference]

## Result
FAIL — TT silicon PCC=0.9818 vs required=0.99; CPU BF16/FP32=0.9935 (above threshold), so precision gap is a compiler issue (consteval-on-host, tt-xla #1242), Tier B

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
consteval-on-host-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9817868226524747. Required: pcc=0.99.

(Original error prior to loader fixes: RuntimeError from Params4bit transfer to non-CUDA device; then RuntimeError: Value out of range for XLA slice start -511)

## Root cause

Three issues found and fixed:

1. **Loader: missing bitsandbytes dependency** — `gemma3_bnb_4bit/pytorch/requirements.txt` did not exist. The pretrained model `unsloth/gemma-3-270m-unsloth-bnb-4bit` loads with `Linear4bit` layers, requiring bitsandbytes.

2. **Loader: BNB Params4bit not transferable to TT device** — `Linear4bit` layers store weights as `Params4bit` tensors which cannot be moved to non-CUDA devices. Fixed by adding `_dequantize_bnb_4bit()` which replaces all `Linear4bit` layers with standard `nn.Linear` using `bnb.functional.dequantize_4bit(weight.data, weight.quant_state)`.

3. **Loader: missing chat_template guard** — The 270M base variant has no chat template but `load_inputs` unconditionally called `apply_chat_template`, raising `ValueError`. Fixed by guarding with `if self.tokenizer.chat_template is not None`.

4. **Tier A fix (tt-xla): XLA slice out-of-range start** — After loader fixes, `DynamicSlidingWindowLayer.update()` produces `full_states[:, :, -sliding_window+1:, :]` with `start = -(512-1) = -511`. For a 256-length sequence, XLA rejects `-511` (valid range `[-256, 255]`), raising `RuntimeError: Value out of range`. Fixed by adding `clamp_out_of_range_slice_starts` FX pass in `tt-xla/python_package/tt_torch/backend/passes.py`, which clamps constant negative starts that exceed `-dim_size` to `-dim_size`. This is semantically correct: both `-511` and `-256` select all elements when `seq_len=256`.

5. **Remaining: Tier B precision bug** — After all fixes, the test runs on silicon but achieves PCC=0.9818 vs required=0.99. CPU BF16 vs CPU FP32 PCC = 0.9935 (above threshold), confirming the BF16 precision floor is NOT the cause. TT silicon has an additional ~0.012 PCC gap beyond what BF16 inherently introduces. This is the consteval-on-host precision bug (tt-xla #1242) where constants (e.g. RoPE embeddings, attention masks) are evaluated at FP32 on host but fed into BF16 computation on device in a way that degrades precision.

## Fix
- `tt_forge_models/gemma3_bnb_4bit/pytorch/requirements.txt` (CREATED): added `bitsandbytes>=0.46.1`
- `tt_forge_models/gemma3_bnb_4bit/pytorch/loader.py` (MODIFIED): added `_dequantize_bnb_4bit()` helper; guard `apply_chat_template` call; call `_dequantize_bnb_4bit(model)` after `from_pretrained`
- `tt-xla/python_package/tt_torch/backend/passes.py` (MODIFIED): added `clamp_out_of_range_slice_starts` FX pass
- `tt-xla/python_package/tt_torch/backend/backend.py` (MODIFIED): import and call `clamp_out_of_range_slice_starts` in `torch_pass_pipeline`

**Proposed fix for Tier B (not attempted):** Investigate and fix the consteval-on-host precision path in tt-xla (issue #1242) to ensure constants evaluated on host preserve precision when fed into BF16 kernels on TT hardware.

## Tier B justification
Which indicator: **cross-cutting**

The consteval-on-host precision issue affects constant evaluation across all model compilations. Fixing it requires understanding and changing how constants (weights materialized on host before device transfer, RoPE tables, mask tensors) are handled through the full compilation pipeline. This is not scoped to one named function and touches the fundamental constant-folding and device-transfer path.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~110s (test ran on silicon, returned PCC=0.9818)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/gemma3_bnb_4bit/pytorch/requirements.txt` (created)
- `tt_forge_models/gemma3_bnb_4bit/pytorch/loader.py` (modified)
- `tt-xla/python_package/tt_torch/backend/passes.py` (modified)
- `tt-xla/python_package/tt_torch/backend/backend.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8df3789b770063c4564c0f054535d4f0d12a5620 |
| tt-forge-models | a94f3eadc8f7c0a8173357abc223cd22f397c3b4 |
