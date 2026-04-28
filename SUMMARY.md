# Remediation Summary: fastvlm_bf16_mlx/image_to_text/pytorch-fastvlm_0_5b_bf16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fastvlm_bf16_mlx/image_to_text/pytorch-fastvlm_0_5b_bf16-single_device-inference]

## Result
FAIL — loader NaN fixed; residual pcc=0.836 is WH BF16 matmul precision floor (Tier B)

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
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.95.
```

(On silicon the required PCC was 0.99; original CI message quoted 0.95.)

## Root cause

**Two separate bugs:**

### Bug 1 — Loader (fixed): mismatched checkpoint keys and MLX NHWC tensor layout

`mlx-community/FastVLM-0.5B-bf16` is a community-converted checkpoint saved under a
different attribute hierarchy than the `llava_qwen.py` code (bundled via `trust_remote_code`)
expects.  `AutoModelForCausalLM.from_pretrained` matched zero weight keys (0/879), so the
language model embeddings, all transformer layers, and the entire vision encoder were left
randomly initialized, producing all-NaN logits.

Key-name mismatches:
- `language_model.model.X` → model expects `model.X`
- `mm_projector.X` → model expects `model.mm_projector.X`
- `vision_tower.vision_model.patch_embed.blocks.N.X` → model expects
  `model.vision_tower.vision_tower.model.patch_embed.N.X`
- `vision_tower.vision_model.X` → model expects
  `model.vision_tower.vision_tower.model.X`

Additionally, all vision-tower conv/scale tensors in the MLX checkpoint are stored in NHWC
format (`(out_c, kH, kW, in_c)` for 4-D, `(1, 1, C)` for 3-D layer-scale), while PyTorch
Conv2d expects NCHW (`(out_c, in_c, kH, kW)` and `(C, 1, 1)` respectively).

### Bug 2 — tt-mlir (unfixed): WH BF16 matmul precision floor

After the loader fix, the test runs to completion with pcc=0.836 vs the required 0.99.
CPU BF16 vs CPU FP32 PCC is 0.9986, confirming BF16 rounding is not responsible for the
gap.  The divergence originates in the TT Wormhole hardware's BF16 matrix-multiply units,
which do not perform FP32 intermediate accumulation.  Accumulated across the 24-layer Qwen2
language model plus the MobileOne + Transformer vision encoder (~585 weight tensors), the
error reaches ~0.164 PCC units below the CPU reference — worse than the 4B-class models
(Qwen3 4B: 0.864, Gemma 7B: 0.915) because the vision encoder adds additional BF16 matmul
operations on top of the LM stack.

## Fix

### Loader fix (applied, branch `remediation/fastvlm_bf16_mlx-image_to_text-pytorch-fastvlm_0_5b_bf16-single_device-inference` in tt-forge-models):

`fastvlm_bf16_mlx/image_to_text/pytorch/loader.py` — replaced `AutoModelForCausalLM.from_pretrained` with:
1. Build the model from config: `AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=True)`
2. Load the safetensors file manually via `hf_hub_download` + `safetensors.torch.load_file`
3. Apply `_remap_checkpoint_keys()`: remaps all four key-prefix patterns listed above, and
   permutes 4-D vision tensors with `permute(0,3,1,2)` and 3-D scale tensors with
   `permute(2,0,1)` to convert from MLX NHWC to PyTorch NCHW
4. `model.load_state_dict(remapped_sd, strict=False)` + `model.tie_weights()`

After the fix: CPU forward pass produces valid logits (no NaN); CPU BF16 vs FP32 PCC = 0.9986.

### Compiler fix (proposed, not attempted):

In tt-mlir, switch BF16 matrix-multiply operations to use HiFi4 accumulation mode
(FP32 intermediate accumulation), or selectively promote matmul inputs to FP32 for
models flagged with a high-fidelity hint.  This is the same fix required for Qwen3 4B and
Gemma 7B.

## Tier B justification
**cross-cutting** — The BF16 matmul precision floor affects every model that runs BF16
matmuls on WH hardware.  Fixing it requires changing the math-fidelity setting for all
matmul operations in the tt-mlir lowering pipeline or adding FP32-promotion passes, which
touches many files across tt-mlir and tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    322.17s (0:05:22) for post-fix run
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/fastvlm_bf16_mlx/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9ec18b23b90f18ef3509136f522f4ad4dfc17204 |
| tt-forge-models | 6475fac5cb5dddfef643ea5d049bd8d374d39d52 |
