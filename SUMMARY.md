# Remediation Summary: llama3_8b_1_58_100b_tokens_gguf-causal_lm-pytorch-Llama3_8B_1_58_100B_tokens_TQ1_0_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama3_8b_1_58_100b_tokens_gguf/causal_lm/pytorch-Llama3_8B_1_58_100B_tokens_TQ1_0_GGUF-single_device-inference]

## Result
FAIL — compiler-stack PCC degradation (0.9094 vs required 0.95/0.99) after loader bugs fixed; root cause unknown

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-pcc-llama3-tq1_0-ternary-weights

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9094181603693505. Required: pcc=0.95.

## Root cause

Two loader bugs were uncovered and fixed during this remediation, but a compiler-stack precision bug remains:

**Loader bug 1 (fixed):** 26 GGUF model loaders monkey-patch `transformers.modeling_utils.load_gguf_checkpoint` at import time with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` function that lacks `**kwargs`. Transformers 5.x added a `model_to_load` positional keyword argument to `load_gguf_checkpoint`, so any of these patchers poisoned the session, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` for this model. Fix: cherry-pick commit 57caeafc70 from `origin/remediation/anubis-mini-gguf-fix-kwargs-compat` to accept and forward `**kwargs`.

**Loader bug 2 (fixed):** When `AutoTokenizer.from_pretrained` loads from a GGUF file as a `TokenizersBackend`, it does not populate `eos_token` or `pad_token` from the GGUF metadata. The loader's `if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token` assignment was a no-op because `eos_token` was also `None`. Calling the tokenizer with `padding=True` then raised `ValueError: Asking to pad but the tokenizer does not have a padding token`. Fix: explicitly register `<|end_of_text|>` (LLaMA-3 EOS, id 128001) via `add_special_tokens` when `eos_token is None`.

**Compiler-stack bug (unfixed):** After the loader fixes, the model loads and runs on silicon but produces PCC=0.9094 vs the CPU bfloat16 reference. CPU BF16 vs FP32 PCC is 0.9999+, ruling out a bfloat16 precision floor. The model is a standard `LlamaForCausalLM` (vanilla 32-layer LLaMA-3 8B architecture) loaded from GGUF with TQ1_0 (1-bit) quantization, whose weights dequantize to ternary bfloat16 values `{-0.00939941, 0, +0.00939941}`. Other LLaMA-3 8B GGUF models with Q4_K_M quantization (more typical weight distributions) pass with SILICON_PASS. The PCC gap is consistent across all 7 sequence positions, suggesting a systematic error in a module applied uniformly to all positions (final RMSNorm, lm_head, or a per-layer operation triggered by ternary weight distributions). The exact operation causing the failure is not identified; diagnosis requires per-layer activation comparison between CPU and TT outputs.

## Fix

**Loader fixes applied (tt-forge-models):**

Remediation branch: `remediation/llama3_8b_1_58_100b_tokens_gguf-causal_lm-pytorch-Llama3_8B_1_58_100B_tokens_TQ1_0_GGUF-single_device-inference`

Commit 1 (cherry-pick of 57caeafc70): Fix `_patched_load_gguf_checkpoint` in 26 GGUF loaders to accept `**kwargs`.
- Files: `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` and 25 others

Commit 2: Fix tokenizer special token registration in llama3_8b_1_58_100b_tokens_gguf loader.
- File: `llama3_8b_1_58_100b_tokens_gguf/causal_lm/pytorch/loader.py`
- Change: replace `if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token` with a guard that calls `add_special_tokens({'eos_token': '<|end_of_text|>', 'pad_token': '<|end_of_text|>'})` when `eos_token` is None.

**Proposed compiler-stack fix (not attempted):** Identify and fix the specific operation in tt-mlir that produces incorrect numerics for ternary bfloat16 weight tensors. Candidate operations: final RMSNorm, lm_head linear projection, or matmul precision for near-zero operands. Per-layer PCC instrumentation is needed first.

## Tier B justification

Which Tier B indicator applies: `internal-error-unknown-mechanism`

The PCC degradation's exact mechanism is not identified. The only known fact is that other LLaMA-3 8B GGUF models with 4-bit (Q4_K_M) weights pass, while this 1-bit (TQ1_0) model fails. The ternary weight structure (`{-0.0094, 0, +0.0094}`) is the distinguishing characteristic, but which compiler operation mishandles it is unknown. Diagnosis (per-layer activation comparison) must precede any fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    347.33s (5:47)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llama3_8b_1_58_100b_tokens_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` (and 25 other GGUF loaders)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | db2e710d6badd8b8264c56f9cd165e0df358f523 |
