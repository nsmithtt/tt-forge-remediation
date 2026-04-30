# Remediation Summary: gemma3_bnb_4bit-1B_Instruct_Unsloth_BNB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_bnb_4bit/pytorch-1B_Instruct_Unsloth_BNB_4bit-single_device-inference]

## Result
FAIL — TT silicon PCC=0.9314 vs required=0.99; same ttmlir-bf16-matmul-precision-floor Tier B bug that affects all Gemma3 models (see tt-xla #3860); BNB quantization+dequantization noise compounds with TT BF16 precision issues

## Stack layer
loader, tt-xla

## Tier
B

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
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`

(Original failure — prior to loader fixes. After loader fixes, test runs on silicon and fails with:
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.931424225241769. Required: pcc=0.99.)

## Root cause

Three loader issues found and fixed:

1. **Loader: missing bitsandbytes dependency** — `gemma3_bnb_4bit/pytorch/requirements.txt` did not exist. The model `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` is stored in BNB NF4 4-bit quantized format and requires `bitsandbytes>=0.46.1` to load. Without it, `AutoModelForCausalLM.from_pretrained` raises `ImportError`.

2. **Loader: BNB Params4bit not transferable to TT device** — After `from_pretrained` with `device_map="cpu"`, all projection layers are `bnb.nn.Linear4bit` with `Params4bit` weight tensors. These cannot be moved to the TT device. Fixed by `_dequantize_bnb_4bit()` which uses `bnb.functional.dequantize_4bit(weight.data, weight.quant_state)` to obtain standard `bfloat16` tensors, then replaces all `Linear4bit` layers with `torch.nn.Linear`.

3. **Loader: chat_template guard** — Added guard so `apply_chat_template` is only called when the tokenizer has a chat template, matching the pattern established for the 270M variant.

**XLA slice fix already present:** The XLA slice out-of-range fix (`clamp_out_of_range_slice_starts` for both `aten.slice.Tensor` and `__getitem__` dispatch paths, for Gemma3's `DynamicSlidingWindowLayer` with `sliding_window=1024` and `seq_len=256`) was already committed to the working branch (commits ee94c31a4 and 9b2a881cf in tt-xla).

**Remaining Tier B — precision:** After all loader fixes, the test runs on silicon (212.72s) but achieves PCC=0.9314 vs required=0.99. The dequantized model weights are in bfloat16 (NF4 dequantize returns bfloat16). CPU FP32 vs CPU BF16 gives PCC=1.0000 (both use BF16 weights after dequantization, confirming the issue is not accumulated BF16 rounding on CPU). The PCC gap is entirely on TT silicon, matching the same `ttmlir-bf16-matmul-precision-floor` Tier B bug that affects regular Gemma3-1B (which gets PCC=0.9558 and requires `assert_pcc: false`, see tt-xla #3860). The BNB quantization→dequantization cycle introduces additional weight noise that compounds with TT's BF16 matmul precision issues, resulting in a larger gap (0.9314 vs 0.9558 for clean weights).

## Fix
- `tt_forge_models/gemma3_bnb_4bit/pytorch/requirements.txt` (CREATED): added `bitsandbytes>=0.46.1`
- `tt_forge_models/gemma3_bnb_4bit/pytorch/loader.py` (MODIFIED): added `_dequantize_bnb_4bit()` helper using `bnb.functional.dequantize_4bit`; chat_template guard in `load_inputs()`; call `_dequantize_bnb_4bit(model)` after `from_pretrained`
- `tt-xla/python_package/tt_torch/torch_overrides.py`: XLA slice clamp fix already in branch (commits ee94c31a4, 9b2a881cf)

**Proposed fix for Tier B (not attempted):** Same fix as for all Gemma3 BF16 models — resolve the `ttmlir-bf16-matmul-precision-floor` issue tracked in tt-xla #3860. Until then, a test config entry with `assert_pcc: false` could be added for this model (consistent with `gemma3/causal_lm/pytorch-1B_Instruct-single_device-inference`).

## Tier B justification
Which indicator: **cross-cutting**

The ttmlir-bf16-matmul-precision-floor issue causes systematic PCC degradation across all BF16 Gemma3 models on TT silicon. Fixing it requires changes to the BF16 matmul lowering path that spans multiple passes and files in the compiler stack. This is not a single scoped fix in one named function.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 212.72s (0:03:32)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gemma3_bnb_4bit/pytorch/requirements.txt` (created)
- `tt_forge_models/gemma3_bnb_4bit/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c2b613d72563b23eb1518e33ca497a7d47bdd770 |
| tt-forge-models | fc2097e5c525a082d1888ab57088603b16617c0b |
