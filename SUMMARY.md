# Remediation Summary: medllama_13b-causal_lm-pytorch-MedLLaMA_13B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medllama_13b/causal_lm/pytorch-MedLLaMA_13B-single_device-inference]

## Result
FAIL — tokenizer RecursionError fixed; residual PCC 0.879 (required 0.99) is a Tier B TT compiler precision bug

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-llama13b-deep

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 23:37:51.091 | critical | Always | TT_FATAL: Chip 0 logical eth core (x=0,y=8) connects to a remote mmio device (assert.hpp:104)

The TT_FATAL ethernet message is hardware-init noise (handled with "Skipping ethernet core" warning). The actual failure on reproduction is:

```
RecursionError: maximum recursion depth exceeded
```

in `AutoTokenizer.from_pretrained` for `chaoyi-wu/MedLLaMA_13B`. After fixing the RecursionError the test fails with:

```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.8789803955103048. Required: pcc=0.99.
```

## Root cause

**Layer 1 — loader (fixed):** The `tokenizer_config.json` for `chaoyi-wu/MedLLaMA_13B`
stores all three special tokens (`bos_token`, `eos_token`, `unk_token`) as empty strings
`""`. In transformers 5.x `LlamaTokenizer.__init__` (now backed by HuggingFace
tokenizers BPE, not SentencePiece), `super().__init__()` triggers
`update_post_processor()` before the backend is fully initialised.
`bos_token_id` → `convert_tokens_to_ids("")` → `token_to_id("")` returns `None` →
`unk_token_id` → same path → infinite recursion.

**Layer 2 — tt-mlir (unfixed):** After applying the loader fix, the model runs to
completion but yields PCC = 0.879 vs CPU. CPU BF16 vs CPU FP32 measures PCC = 0.9995,
ruling out BF16 accumulation as the cause. MedLLaMA-13B is 40 layers with 5120 hidden
dimension (full MHA, 40 heads, original LLaMA-1 architecture). The divergence between
TT and CPU is compiler-origin but the specific mechanism is unknown: candidates include
SDPA mask handling for heavily-padded sequences (4 real tokens / 124 padding out of 128),
or precision loss in the 5120-wide linear projections across 40 layers.

## Fix

**Loader fix (applied):** In `medllama_13b/causal_lm/pytorch/loader.py`, pass
`bos_token="<s>"`, `eos_token="</s>"`, `unk_token="<unk>"` explicitly to
`AutoTokenizer.from_pretrained`. These kwargs override the empty-string values from the
checkpoint's `tokenizer_config.json` via `init_kwargs.update(kwargs)` in `_from_pretrained`,
allowing `token_to_id` to resolve the special tokens correctly.

**Compiler fix (proposed, not attempted):** Diagnose why TT produces PCC 0.879 for
LLaMA-1 13B (40 layers, 5120 hidden). Likely investigation starting points:
- SDPA attention-mask handling when most of the sequence is padding tokens
  (`src/api/module_builder/frontend_passes/` in tt-xla, or SDPA lowering in tt-mlir)
- BF16 precision in 5120-wide matmul ops across 40 transformer layers

## Tier B justification
**cross-cutting** — The PCC failure spans all 40 transformer layers of LLaMA-13B.
The root cause is unknown; multiple ops (SDPA, matmul, RMSNorm, RoPE) are candidates.
Without per-layer diagnosis, the mechanism cannot be isolated to a single-file fix.
CPU BF16 gives PCC 0.9995 vs FP32, confirming the bug is TT-specific and not inherent
to BF16 arithmetic.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    244.73s (0:04:04)
- Tier A attempts: 0

## Files changed
- `medllama_13b/causal_lm/pytorch/loader.py` (modified: special-token override)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f18949ea8d857d05a486fcf7727d5b6fe0fef0ff |
| tt-forge-models | b08fc4c9f82a93caf981886333f6628d896c47a0 |
