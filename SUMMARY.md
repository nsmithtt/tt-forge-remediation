# Remediation Summary: glm_4v-pytorch-cogagent_9b_20241220-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4v/pytorch-cogagent_9b_20241220-single_device-inference]

## Result
FAIL — loader bug fixed (attention_mask.all() TT device-to-host Error code 13); remaining Tier B compiler-stack PCC failure (pcc=0.7369, required=0.99)

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Error code: 13

Full traceback:
  modeling_chatglm.py:1251: in forward
      if (attention_mask is not None and not attention_mask.all()) or (
                                             ^^^^^^^^^^^^^^^^^^^^
  python_package/tt_torch/torch_overrides.py:34: in __torch_function__
      return func(*args, **(kwargs or {}))
  E   RuntimeError: Error code: 13

After loader fix: PCC comparison failed. Calculated: pcc=0.7369988920072369. Required: pcc=0.99.

## Root cause
Two distinct issues:

**Issue 1 (loader, fixed):** `ChatGLMModel.forward` (modeling_chatglm.py:1251) evaluates
`attention_mask.all()` to decide whether to build an explicit causal mask. On TT tensors this
requires a device-to-host transfer, which the PJRT plugin cannot perform and raises INTERNAL
Error code: 13. For unpadded single-prompt inputs the attention mask is always all-ones, so
the condition evaluates to False in either case — removing the mask makes the check unreachable
without changing model behavior.

**Issue 2 (compiler stack, unfixed):** After the loader fix the model runs to completion but
produces PCC 0.7369 vs required 0.99. The model processes a 1120×1120 image through a ViT
vision encoder (1600 patches) before passing the combined image+text embeddings through the
text transformer. PCC 0.74 is far below the expected BF16 floor (~0.95+) and indicates a real
computation error, not just floating-point rounding. The most likely candidate is an op
lowering issue in the vision encoder (conv layers, patch embedding, or ViT attention at
seq_len=1600), but the specific failing op has not been isolated.

## Fix
**Loader fix (applied):** In `glm_4v/pytorch/loader.py` `load_inputs()`, added a check that
removes `attention_mask` from the inputs dict when the mask is all-ones. With `attention_mask=None`
the model's `if attention_mask is not None and not attention_mask.all()` condition short-circuits
at the `is not None` check, bypassing the device-to-host transfer entirely. The `SdpaAttention`
forward then uses `is_causal=True`, which produces identical outputs to an explicit all-ones causal mask.

**Compiler fix (proposed, not implemented):** The PCC issue requires isolating which op in the
vision encoder (likely a conv, layer norm, or SDPA at 1600-token sequence length) produces
incorrect results on TT hardware, then applying a targeted lowering fix in tt-mlir.

## Tier B justification
`internal-error-unknown-mechanism` — The PCC of 0.7369 indicates a real computation error
but the specific op causing it has not been identified. Root-cause diagnosis of the vision
encoder on TT hardware (which involves isolating among conv, attention, and MLP ops in the ViT
over 1600 patches) requires investigation first. Cross-cutting if both vision and text
transformer ops are affected.

## Verification
- pytest exit: FAIL (PCC 0.7369 < 0.99 after loader fix)
- Hardware:    blackhole-p150b
- Duration:    406.62s (0:06:46)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/glm_4v/pytorch/loader.py` — remove all-ones attention_mask in load_inputs

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 72ad371423f (remediation/glm_4v-pytorch-cogagent_9b_20241220-single_device-inference) |
