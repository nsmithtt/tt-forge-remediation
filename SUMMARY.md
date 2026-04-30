# Remediation Summary: gemma3_logiqa_dpo_lora-pytorch-4B_Instruct_LogiQA_DPO-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_logiqa_dpo_lora/pytorch-4B_Instruct_LogiQA_DPO-single_device-inference]

## Result
FAIL — three loader bugs fixed; silicon PCC=0.9106 is the WH BF16 matmul precision floor (Tier B)

## Stack layer
loader, tt-mlir

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
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

The reported failure detail was only the last line of pytest stderr (a SWIG deprecation
warning). The three actual loader failures and the terminal silicon PCC failure were:

1. GatedRepoError: `google/gemma-3-4b-it` is HuggingFace-gated; CI machine had the
   model cached from a prior authorized run. Locally reproduced after switching base
   model to `unsloth/gemma-3-4b-it`.

2. TypeError: `Gemma3ForConditionalGeneration.__init__() got an unexpected keyword
   argument 'use_cache'` — transformers 5.x removed `**kwargs` from the Gemma3
   multimodal constructor; `use_cache=False` must be set via `config.text_config`.

3. ValueError: `not enough values to unpack (expected 4, got 2)` —
   `Gemma3ForConditionalGeneration.forward` signature is
   `(input_ids, pixel_values, attention_mask, ...)`. Returning inputs as positional
   args caused `attention_mask` to bind to `pixel_values`. Fix: return a dict from
   `load_inputs()` so inputs are passed as keyword args.

4. Terminal: `AssertionError: PCC comparison failed. Calculated: pcc=0.9105699202627927.
   Required: pcc=0.99.`

## Root cause
Three cascading loader bugs, then a Tier B compiler precision floor:

**Bug 1 (loader)**: `BASE_MODEL_NAME = "google/gemma-3-4b-it"` is a gated HF repo.
A prior commit (`922141180e`) had already switched to `unsloth/gemma-3-4b-it`, but
merge commit `a01439d375` (merging `origin/nsmith/hf-bringup`) reverted the file
back to the gated name, reintroducing the bug.

**Bug 2 (loader)**: transformers 5.x tightened the `Gemma3ForConditionalGeneration`
constructor to accept only `config` (no `**kwargs`). The loader was passing
`use_cache=False` via `model_kwargs` which `from_pretrained` forwards verbatim to
`cls(config, **model_kwargs)`. In transformers 4.x this was silently ignored.

**Bug 3 (loader)**: `Gemma3ForConditionalGeneration.forward` places `pixel_values` as
the second positional parameter before `attention_mask`. The loader returned
`[input_ids, attention_mask]` as a list (positional args), so `attention_mask` mapped
to `pixel_values` instead of its correct slot. The model then tried to unpack the 2D
attention mask as a 4D image tensor `(batch, channels, height, width)`.

**Tier B — ttmlir-bf16-matmul-precision-floor**: After all loader fixes, the model
runs on silicon but produces PCC=0.9106, well below the required 0.99. This matches
the known WH BF16 matmul accumulation error in tt-mlir for deep models in the Gemma
family (also observed at PCC≈0.915 for Gemma 7B, PCC=0.864 for Qwen3 4B). Gemma 3
4B-IT has 34 transformer layers; BF16 rounding error in the TTNN matmul kernel
compounds across layers. Preserving F32 precision through lowering is a cross-cutting
change (Tier B).

## Fix
**Three loader fixes in `tt-forge-models`, branch
`remediation/gemma3_logiqa_dpo_lora-pytorch-4B_Instruct_LogiQA_DPO-single_device-inference`:**

1. `gemma3_logiqa_dpo_lora/pytorch/loader.py` — change `BASE_MODEL_NAME` from
   `"google/gemma-3-4b-it"` to `"unsloth/gemma-3-4b-it"` (ungated mirror,
   architecturally identical).

2. Same file — replace `model_kwargs = {"use_cache": False}` with loading the config
   and setting `config.text_config.use_cache = False`, then passing `config=config`
   in `model_kwargs`. Also replace deprecated `torch_dtype=` with `dtype=`.

3. Same file — change `load_inputs()` return from
   `[inputs["input_ids"], inputs["attention_mask"]]` to
   `{"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}` so
   inputs are bound by keyword, leaving `pixel_values=None` for text-only inference.

**Proposed fix for Tier B (not implemented)**: The `ttmlir-bf16-matmul-precision-floor`
bug would require changing TTNN matmul accumulation precision from BF16 to F32 across
all lowering passes in tt-mlir, or introducing per-op precision hints. This is a
cross-cutting change touching many files across tt-mlir and tt-metal.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting — preserving F32 precision through every matmul lowering in tt-mlir
requires coordinated changes across multiple passes and files; this is the same
`ttmlir-bf16-matmul-precision-floor` Tier B seen for Gemma 7B (tt-xla #2861) and
Qwen3 4B.

## Verification
- pytest exit: FAIL (PCC=0.9106, required=0.99)
- Hardware:    n150
- Duration:    203.43s (0:03:23)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/gemma3_logiqa_dpo_lora/pytorch/loader.py` (3 commits)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c2b613d72563b23eb1518e33ca497a7d47bdd770 |
| tt-forge-models | d01055d36a7ab6d5c17fddf978a987fe265e915c |
