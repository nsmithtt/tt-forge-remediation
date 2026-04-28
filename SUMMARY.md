# Remediation Summary: acestep_v15_base-pytorch-base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[acestep_v15_base/pytorch-base-single_device-inference]

## Result
FAIL — loader meta-tensor bug fixed; residual PCC failure (0.9801 < 0.99) is a Tier B WH BF16 matmul precision issue

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
wh-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Tensor.item() cannot be called on meta tensors

(Original error, from vector_quantize_pytorch/residual_fsq.py:84 `assert (levels_tensor > 1).all()`)

## Root cause

Two bugs found in sequence:

**Bug 1 (loader, fixed):** `transformers 5.x` unconditionally wraps model construction in
`torch.device("meta")` via `PreTrainedModel.get_init_context()`. The ACE-Step model uses
`trust_remote_code=True` and its `AceStepAudioTokenizer` creates a `ResidualFSQ` (from the
`vector_quantize_pytorch` package) during `__init__`. Both `ResidualFSQ.__init__`
(`assert (levels_tensor > 1).all()`) and `FSQ.__init__`
(`self.codebook_size = self._levels.prod().item()`) call `.all()` / `.item()` on tensors
constructed from Python lists. Inside the meta-device context, `torch.tensor([2, 4, 8])`
creates meta tensors, and `.all()` / `.item()` on meta tensors raises
`RuntimeError: Tensor.item() cannot be called on meta tensors`.

Fix: temporarily remove `torch.device("meta")` from the contexts returned by
`get_init_context` so that `ResidualFSQ` / `FSQ` construct on CPU.

**Bug 2 (tt-mlir, unfixed):** After the loader fix, the model loads and runs on TT silicon
but produces PCC=0.9801 (required: 0.99). The reference CPU BF16 vs FP32 PCC is 0.9999,
so the 0.019 gap is not explained by BF16 accumulation in the model itself. This matches
the known WH BF16 matmul precision issue that compounds across layers. The ACE-Step DiT
decoder has `hidden_size=2048`, `intermediate_size=6144`, 24 layers — similar in scale to
Gemma 7B (PCC~0.915, Tier B) and Qwen3 4B (PCC=0.864, Tier B). The precision floor is
better here (0.9801) because the model has fewer layers, but it is the same root cause:
WH BF16 matmul accumulation error tracked in tt-xla #2861.

## Fix

**Loader fix applied** (`tt-forge-models`):
- `acestep_v15_base/pytorch/loader.py`: patch `PreTrainedModel.get_init_context` in a
  `try/finally` block to filter out `torch.device` entries before calling
  `AutoModel.from_pretrained`, then restore the original method.

**Compiler fix not attempted (Tier B):**
The residual PCC failure requires preserving FP32 precision through the tt-mlir lowering
passes for matmul-heavy models on Wormhole hardware. This is a cross-cutting change.

## Tier B justification
cross-cutting — the fix requires preserving f32 precision through every matmul lowering
pass in tt-mlir for WH hardware, matching the pattern of tt-xla #2861 (same as Gemma 7B
and Qwen3 4B Tier B reports).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    224.63s (0:03:44)
- Tier A attempts: N/A

## Files changed
- `acestep_v15_base/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 65894daaf5a7a6922ee36c124ae13d7b5d93df9d |
| tt-forge-models | 2ded63955e839896d7a98186ed678bb2ce060138 |
