# Remediation Summary: mochi_comfyui_repackaged-pytorch-Preview_VAE-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mochi_comfyui_repackaged/pytorch-Preview_VAE-single_device-inference]

## Result
FAIL — SDPA with non-tile-aligned temporal sequence lengths (7, 4, 2) gives PCC=0.55 (Tier B ttnn-sdpa-nonaligned-kv-pcc-wrong)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.5513488029389646. Required: pcc=0.99.

## Root cause
The Mochi VAE uses causal temporal attention (MochiVaeAttnProcessor2_0) in all its encoder and decoder blocks. The attention operates over the temporal dimension after spatially flattening: shape [B×H×W, heads, T, head_dim] where T ∈ {7, 4, 2} depending on the block.

All 16 SDPA calls in the full VAE encode-decode pass use non-tile-aligned sequence lengths (none is a multiple of 32). The TT SDPA kernel produces wrong values for non-tile-aligned K/V sequence lengths, causing PCC=0.55 vs the CPU reference.

CPU BF16 vs FP32 PCC = 0.9992 (baseline healthy), confirming this is not a precision floor but a real compiler bug.

The loader had two prior bugs (from the hf-bringup branch commits af6ebbf2c7 and ce8ed4edbd):
1. `from_single_file` not available on AutoencoderKLMochi in diffusers 0.37.1 — fixed by using `from_pretrained("genmo/mochi-1-preview", subfolder="vae")`.
2. `load_inputs` used `**kwargs` making `dtype_override` invisible to `inspect.signature` — fixed by adding it as an explicit named parameter.

After these loader fixes, the test reaches silicon and exposes the SDPA bug.

## Fix
Loader fix in tt_forge_models on branch `remediation/mochi_comfyui_repackaged-pytorch-Preview_VAE-single_device-inference` (commit 961cbc33cb):
- `mochi_comfyui_repackaged/pytorch/loader.py`: changed `def load_inputs(self, **kwargs)` to `def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs)` so `dtype_override` is detected by `inspect.signature` and bfloat16 inputs are passed.

The SDPA bug (tt-mlir) requires padding K/V to the next tile-aligned size (multiple of 32) and applying appropriate masking. This is the same bug previously observed in Chronos2 (seq_len=34) and other models.

## Tier B justification
new-infrastructure

Padding K/V to tile alignment in the SDPA lowering path requires new infrastructure: dynamically inserting pad ops for the sequence dimension, computing a pad mask, and broadcasting it alongside the optional user mask. This is not a scoped one-file change — it requires coordinated changes across the SDPA verifier, the padding/masking logic, and the TTIR→TTNN lowering patterns.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    391.69s (0:06:31)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: `mochi_comfyui_repackaged/pytorch/loader.py` — add `dtype_override` as explicit named parameter to `load_inputs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4136032e6e543883d7aafd2fef50d06b1cde4a4f |
| tt-forge-models | 961cbc33cb85625de66adf1bedf3dfbde24a7eba |
