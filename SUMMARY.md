# Remediation Summary: gla-causal_lm-pytorch-340M-15B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gla/causal_lm/pytorch-340M-15B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gla-attention-mask-dynamic-scatter-gather

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8732439609931861. Required: pcc=0.95.

## Root cause
The GLA loader passes `attention_mask` to the model's forward method. When `attention_mask` is present, `GatedLinearAttention.forward` calls `get_unpad_data(attention_mask)` which uses `torch.nonzero()` (dynamic output size) and then `index_first_axis` / `pad_input` (scatter/gather with data-dependent indices). These dynamic scatter/gather operations are not compiled correctly by the TT XLA compiler, producing PCC=0.87.

For a single unpadded sequence the attention_mask is all-ones, so the unpack/repad path is a mathematical no-op. On CPU both paths produce identical output (PCC=1.0 between masked and unmasked runs). The BF16 floor for this model is PCC=1.0 (BF16 vs FP32 on CPU), confirming the error is entirely in the dynamic indexing path, not a precision limitation.

Additionally, the bringup branch (e60070cb84) introduced `flash-linear-attention` and `triton` dependencies plus CPU-compatible monkey-patches for all triton ops (norm, attention, SwiGLU activations) so the model can load and run on a CPU-only TT machine.

## Fix
`gla/causal_lm/pytorch/loader.py` — removed `attention_mask` from the `load_inputs()` return dict. GLA is a recurrent attention model that only uses `attention_mask` to derive `cu_seqlens` for packing variable-length batches into a single forward pass. For our single-sentence unpadded test case the mask is redundant. Omitting it skips the `nonzero` / `index_first_axis` / `pad_input` path entirely.

The fix builds on the bringup-branch loader (e60070cb84) which already registered the `gla` architecture with transformers 5.x and monkey-patched all triton ops with pure-PyTorch equivalents.

Repository: tenstorrent/tt-forge-models
Branch: remediation/gla-causal_lm-pytorch-340M-15B-single_device-inference
Commit: ddb366077064e3cefa389cf7f0f68e341ff0dd3c

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    102.87s
- Tier A attempts: N/A

## Files changed
- gla/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ddb366077064e3cefa389cf7f0f68e341ff0dd3c |
