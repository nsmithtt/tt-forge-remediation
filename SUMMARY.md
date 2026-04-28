# Remediation Summary: pythia-causal_lm-pytorch-h2ogpt-oig-oasst1-256-6_9b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[pythia/causal_lm/pytorch-h2ogpt-oig-oasst1-256-6_9b-single_device-inference]

## Result
FAIL — BF16 accumulation precision loss across 32 decoder layers gives PCC=0.9439, below the 0.95 CI threshold; root cause is ttmlir-f32-precision-not-preserved (Tier B, cross-cutting)

## Stack layer
tt-mlir

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9439138536262934. Required: pcc=0.95.

## Root cause
The logits tensor (evaluation result 0) has PCC=0.9439 between TT silicon and CPU. The h2ogpt-oig-oasst1-256-6_9b model is a fine-tune of EleutherAI/pythia-6.9b, which is a GPT-NeoX architecture with 32 decoder layers and 4096 hidden dimension. Running in bfloat16, TT hardware accumulates matmul results in bfloat16 while CPU PyTorch (via MKL) accumulates in float32 internally. Over 32 decoder layers each containing 3–4 large matmul operations (QKV projection, output projection, MLP up/down), the accumulated bfloat16 rounding error compounds to a ~5.6% PCC degradation. This is the same bug previously identified for Whisper Large v3 Turbo (36 layers, 1280 hidden → pcc≈0.94) and other deep bfloat16 models.

No loader-level issues were found: the tokenizer uses `padding=True` (no padding for a single input), input shapes are natural (≈50 tokens, no truncation), and `generation_config` in the inputs dict is absorbed by `**kwargs` in `GPTNeoXForCausalLM.forward` and has no effect on logits computation. The `torch_dtype` deprecation warning from transformers 5.x is benign — the bfloat16 dtype is still applied correctly.

## Fix
No fix attempted; Tier B.

The underlying fix would require tt-mlir to lower bfloat16 matmul operations to use float32 accumulation (or mixed-precision accumulation) throughout the StableHLO→TTIR→TTNN pipeline. This is a cross-cutting change touching every matmul lowering in tt-mlir and tt-metal kernels — well beyond the scope of a single-PR Tier A fix.

## Tier B justification
cross-cutting — preserving float32 accumulation through every matmul lowering pass requires changes to tt-mlir op lowerings and tt-metal kernel configurations; it is not scoped to one function or one file.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    155.26s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
