# Remediation Summary: gpt_j_6b_janeway-causal_lm-pytorch-default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_j_6b_janeway/causal_lm/pytorch-default-single_device-inference]

## Result
FAIL — PCC=0.7487 after loader fix; WH BF16 matmul precision error in GPT-J large MLP (16384-dim inner product), Tier B cross-cutting

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.7486711108969377. Required: pcc=0.95.

## Root cause
Two issues were identified:

**Loader bug (fixed):** GPT-J uses the GPT-2 tokenizer, which has no `pad_token` defined. Calling the tokenizer with `padding=True` raises `ValueError: Asking to pad but the tokenizer does not have a padding token`. The original loader also incorrectly passed `torch_dtype` to `AutoTokenizer.from_pretrained()`, which does not accept it. Fixed by setting `tokenizer.pad_token = tokenizer.eos_token` after loading and removing the stray kwarg.

**PCC failure (unfixed, Tier B):** After the loader fix, the model compiles and runs on TT silicon with PCC=0.7487 vs CPU BF16 reference (CPU BF16 vs FP32 PCC=0.9998, confirming the error is entirely in TT execution). GPT-J 6B has `n_inner=None`, meaning `intermediate_size = 4 × n_embd = 16384`. The MLP `fc_out` matmul contracts over 16384 BF16 dimensions per layer. Over 28 decoder layers, Wormhole's BF16 matmul accumulates rounding error proportional to √16384 × BF16_epsilon ≈ 128 × 0.0078 ≈ 1.0 per layer. This compounds to PCC≈0.75, consistent with other known large-model BF16 matmul failures (tt-xla #2861). An alternative routing of GPT-J's `gelu_new` activation through the `tenstorrent.gelu_tanh` composite op was attempted but produced no PCC improvement, ruling out the activation function as a cause.

## Fix
**Applied (loader):** In `tt-xla/third_party/tt_forge_models/gpt_j_6b_janeway/causal_lm/pytorch/loader.py`:
- Set `self.tokenizer.pad_token = self.tokenizer.eos_token` after loading the tokenizer.
- Removed stray `torch_dtype` kwarg passed to `AutoTokenizer.from_pretrained()`.

**Proposed (compiler stack):** Preserve float32 precision through all TTIR/TTNN lowering passes so that large BF16 matmuls are computed with adequate precision. Tracked as tt-xla #2861. This would require changes to the WH BF16 matrix multiplication kernels in tt-metal and associated lowering patterns in tt-mlir.

## Tier B justification
cross-cutting — Wormhole BF16 matmul precision is a hardware-level accumulation characteristic that affects every model using large inner dimensions. A fix requires coordinated changes to tt-metal matrix kernels and tt-mlir lowering patterns across multiple files and modules. Same root cause as the known Gemma 7B / Qwen3 4B Tier B failures (tt-xla #2861).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    154.30s (0:02:34)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gpt_j_6b_janeway/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1e2a552973dc63153f9f6d144c97bcb2a05a72a3 |
| tt-forge-models | 366c1e6b52f5ad42ed9b8576328433ee7dfa8b01 |
