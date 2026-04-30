# Remediation Summary: indic_gemma_gguf-causal_lm-pytorch-7B_SFT_NAVARASA_2_0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[indic_gemma_gguf/causal_lm/pytorch-7B_SFT_NAVARASA_2.0_GGUF-single_device-inference]

## Result
FAIL — Tier B WH BF16 matmul precision floor: PCC=0.9414 < 0.99 required; same root cause as Gemma 7B IT (tt-xla #2861)

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9414031486146744. Required: pcc=0.95.

## Root cause

**Three loader bugs** were blocking the test before reaching silicon; all are fixed. A fourth, Tier B compiler-precision issue remains.

**Bug 1 — Wrong GGUF filename** (loader): The loader hardcoded
`Indic-gemma-7b-finetuned-sft-Navarasa-2.0-i1-Q4_K_M.gguf` (hyphen before `i1`).
The mradermacher repo uses dot notation: `Indic-gemma-7b-finetuned-sft-Navarasa-2.0.i1-Q4_K_M.gguf`.
CI had a cached copy; a clean environment raises `OSError: file not found`.

**Bug 2 — Gemma v1 GGUF arch not in transformers 5.x** (loader): The GGUF stores
`general.architecture='gemma'`. Transformers 5.x dropped gemma v1 from
`GGUF_CONFIG_MAPPING` (only `gemma2`/`gemma3` remain), causing
`ValueError: GGUF model with architecture gemma is not supported yet.`

**Bug 3 — Narrow-sig `load_gguf_checkpoint` wrappers strip `model_to_load`** (loader):
Transformers 5.2+ calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=model)`.
Multiple other loaders in the test suite install narrow-signature wrappers
`(gguf_path, return_tensors=False)` at import time. Since Python's test collection
imports all loaders, these wrappers are active at run time and reject the
`model_to_load` kwarg with a `TypeError`.

**Residual — WH BF16 matmul precision floor** (tt-mlir/tt-metal): After all
loader fixes, the model compiles and executes on silicon with PCC=0.9414031486146744
(identical to the original CI value, confirming the loader bugs masked the precision
failure). This is the same WH BF16 matmul accumulation issue documented for the
base Gemma 7B model in tt-xla #2861. Indic Gemma is a 28-layer fine-tune of
Gemma 7B; both hit the same precision floor. Fix requires f32 accumulation in the
TTIR matmul lowering pass — a cross-cutting change across all lowering stages.

## Fix

**Bug 1** — `indic_gemma_gguf/causal_lm/pytorch/loader.py`: change
`GGUF_FILE = "Indic-gemma-7b-finetuned-sft-Navarasa-2.0-i1-Q4_K_M.gguf"` →
`GGUF_FILE = "Indic-gemma-7b-finetuned-sft-Navarasa-2.0.i1-Q4_K_M.gguf"`.

**Bug 2** — `indic_gemma_gguf/causal_lm/pytorch/loader.py`: add
`_patch_gemma_v1_support()` at module level. Re-registers `gemma` in
`GGUF_TO_TRANSFORMERS_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`,
`TENSOR_PROCESSORS`, and `GGUF_TO_FAST_CONVERTERS` using Gemma 2 as a template.
Also adds a `chat_template is not None` guard in `load_inputs` since Gemma v1
GGUF tokenizers do not embed a chat template.

**Bug 3** — `indic_gemma_gguf/causal_lm/pytorch/loader.py`: add
`_get_real_load_gguf_checkpoint()` which performs a recursive DFS over both
`__closure__` cells and `__globals__` (names containing `orig`) to find the
real `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` (identified
by `__qualname__=='load_gguf_checkpoint'` and
`__module__=='transformers.modeling_gguf_pytorch_utils'`). In `load_model`,
the real function is installed just before `AutoModelForCausalLM.from_pretrained`
and restored in a `finally` block.

**Proposed fix for residual Tier B**: add a test config entry mirroring Gemma 7B IT:
```yaml
indic_gemma_gguf/causal_lm/pytorch-7B_SFT_NAVARASA_2.0_GGUF-single_device-inference:
    status: EXPECTED_PASSING
    assert_pcc: false  # WH BF16 matmul precision floor, same as tt-xla #2861
```
The fix itself requires `fp32_dest_acc_en` or equivalent in the TTIR BF16 matmul
lowering — tracked under tt-xla #2861.

## Tier B justification
cross-cutting: fixing the BF16 precision floor requires preserving f32 accumulation
through every BF16 matmul lowering in tt-mlir (StableHLOToTTIR + TTIRToTTNN
lowering stages), coordinated with tt-metal kernel changes. This affects all models
using BF16 matmuls, not just Gemma.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    489.93s (0:08:09)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/indic_gemma_gguf/causal_lm/pytorch/loader.py`
  (tt-forge-models remediation branch: ae7faf5034)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ae7faf5034c1e3188e6dc1c9dec09f50015a6e20 |
