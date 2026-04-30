# Remediation Summary: doctr_parseq_multilingual-pytorch-parseq-multilingual-v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[doctr_parseq_multilingual/pytorch-parseq-multilingual-v1-single_device-inference]

## Result
FAIL — PCC=0.9194 below required 0.99; WH BF16 matmul precision floor in 12-layer ViT encoder with large residual activations (max≈2672 at layer 11)

## Stack layer
loader, tt-xla, tt-metal

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
2026-04-23 21:54:45.733 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)

The original CI failure was a transient eth-core MMIO warning (known non-fatal, device continues). The actual failure under test is `PCC comparison failed. Calculated: pcc=0.9194638911781048. Required: pcc=0.99.`

## Root cause
Three independent bugs were found and fixed:

1. **Missing dependency** (loader): `python-doctr` was not installed; the loader uses `from doctr.models import from_hub` with no `requirements.txt`.

2. **String output not tensor-comparable** (loader): `PARSeq.forward()` with default args always includes `preds` in the output dict — a list of `(text, confidence)` tuples. `tree_map` over this pytree calls `torch.equal(str, str)` → `TypeError`. Fix: `model.exportable = True` makes `forward()` return `{"logits": tensor}` early, bypassing the string decoder and allowing PCC comparison.

3. **_guards_fn dead node NameError** (tt-xla): Dynamo-generated symbolic-shape guard nodes are injected into the FX graph during `run_decompositions` re-tracing. Their `forward()` closes over `L` (the Dynamo locals dict), absent at inference time → `NameError: name 'L' is not defined`. Fixed by cherry-picking `d55e16661` (skip and erase dead `_guards_fn` nodes).

**Remaining Tier B failure**: After all three fixes, TT gives PCC=0.9194 (required 0.99). Root cause is the WH BF16 matmul precision floor accumulating through 12 layers of the ViT feature extractor:
- Patch embedding TT PCC: 0.999995 (single Conv2d, essentially perfect)
- Full feature extractor TT PCC: 0.956 (12-layer EncoderBlock with pretrained weights)
- BF16-CPU vs FP32-CPU PCC: 0.999157 (not a BF16 floor issue on CPU)
- Single MHA layer TT PCC: 0.999832; single FFN TT PCC: 0.999933 (individual layers fine)

The residual connections in the pre-LN ViT accumulate large activation magnitudes: max≈712 at layer 0, growing to max≈2672 at layer 11. BF16 step size at magnitude 2672 is ≈16–32, giving ≈1% relative precision per operation. Compounded across 12 layers of attention + FFN, the encoder output reaches PCC=0.956. The decoder compounds this further to PCC=0.9194 for the full model.

This is the same `ttmlir-bf16-matmul-precision-floor` seen in Gemma 7B (PCC≈0.915, 32 layers) and Qwen3 4B (PCC=0.864, 36 layers), manifesting here in a smaller ViT with large per-layer activation growth.

## Fix
**Loader fixes** (tt_forge_models, `doctr_parseq_multilingual/pytorch/`):
- `requirements.txt` (new file): `python-doctr>=1.0.1`
- `loader.py`: added `model.exportable = True` after `from_hub(...)` so the model returns `{"logits": tensor}` instead of `{"preds": [(str, float)]}`, enabling PCC comparison

**tt-xla fix** (`python_package/tt_torch/backend/backend.py`):
- Cherry-picked commit `d55e16661` — patch `PropagateUnbackedSymInts.run_node` to skip dead `_guards_fn` call_module nodes during `run_decompositions`, then erase them from the compiled graph

**Proposed fix for Tier B** (tt-metal / tt-mlir):
- Enable FP32 accumulation for BF16 matmuls on Wormhole hardware (TTNNWorkaroundsPass or via math fidelity override), similar to `MATH_FIDELITY_HI_LO` → `MATH_FIDELITY_HI_HI`; cross-cutting change touching all matmul lowerings

## Tier B justification
cross-cutting — enabling F32 accumulation for BF16 matmuls requires coordinated changes across all matmul lowering paths in tt-mlir/tt-metal; cannot be scoped to one file or function

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    113.33s (1:53)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: `doctr_parseq_multilingual/pytorch/requirements.txt` (new)
- tt_forge_models: `doctr_parseq_multilingual/pytorch/loader.py` (exportable=True)
- tt-xla: `python_package/tt_torch/backend/backend.py` (cherry-pick d55e16661)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 81a9cda526531a1bd883dc7ae4cf785910a49216 |
| tt-forge-models | 38821085ee5247151dba88f15acd71eb5046adca |
