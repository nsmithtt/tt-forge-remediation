# Remediation Summary: longcat_video_avatar_comfyui_gguf-pytorch-Single_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longcat_video_avatar_comfyui_gguf/pytorch-Single_Q4_K_M-single_device-inference]

## Result
FAIL — loader uses wrong model class (`WanTransformer3DModel`) for a custom `LongCatVideoAvatarTransformer3DModel` architecture

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-wrong-model-class-longcat-video-avatar

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (reproduced):
```
OSError: stable-diffusion-v1-5/stable-diffusion-v1-5 does not appear to have a file named config.json.
```

Root failure after partial loader fix:
```
ValueError: Cannot load  because blocks.0.attn2.to_k.bias expected shape torch.Size([5120]), but got torch.Size([4096]).
```

## Root cause
Two-layer loader bug in `longcat_video_avatar_comfyui_gguf/pytorch/loader.py`:

**Bug 1 (config detection)**: `WanTransformer3DModel.from_single_file()` calls
`infer_diffusers_model_type(checkpoint)` which checks for the GGUF key
`head.modulation` as a WAN indicator. LongCat's GGUF doesn't have this key
(it uses a custom architecture), so detection falls back to `"v1"` →
`stable-diffusion-v1-5/stable-diffusion-v1-5` → 404.

**Bug 2 (wrong model class)**: The actual model is
`LongCatVideoAvatarTransformer3DModel` (`meituan-longcat/LongCat-Video-Avatar`,
`avatar_single/config.json`), not `WanTransformer3DModel`. Key differences:
- Hidden dim: 4096 (LongCat) vs 5120 (WAN T2V 14B)
- Num layers: 48 vs 40
- FFN: SwiGLU with 3 matrices (w1/w2/w3, intermediate=11008) vs GELU with
  2 matrices (net.0.proj/net.2, intermediate=13824)
- Extra components: `audio_cross_attn`, `audio_modulation` in every block

`LongCatVideoAvatarTransformer3DModel` is not available in diffusers 0.37.1.
The partial fix (GGUFQuantizationConfig + explicit config + subfolder) resolves
Bug 1 but exposes the fundamental architecture incompatibility of Bug 2.

## Fix
**Partial fix committed** to `tenstorrent/tt-forge-models` branch
`remediation/longcat_video_avatar_comfyui_gguf-pytorch-Single_Q4_K_M-single_device-inference`:
- Add `GGUFQuantizationConfig(compute_dtype=compute_dtype)` for proper GGUF handling
- Pass `config="Wan-AI/Wan2.1-T2V-14B-Diffusers"` + `subfolder="transformer"`
  to bypass auto-detection failure

**Proposed complete fix** (not implemented): Rewrite the loader to either:
1. Implement `LongCatVideoAvatarTransformer3DModel` with SwiGLU FFN (w1/w2/w3),
   audio cross-attention blocks, and custom GGUF key mapping — OR —
2. Load the model from the official `meituan-longcat/LongCat-Video-Avatar`
   safetensors with a custom model class implementation

The loader would need to live in `longcat_video_avatar_comfyui_gguf/pytorch/loader.py`
and implement key remapping: GGUF `blocks.N.ffn.w1/w2/w3` → custom FFN modules,
GGUF `blocks.N.audio_cross_attn.*` → audio attention modules.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    ~28s (loader failure, no silicon run)
- Tier A attempts: N/A

## Files changed
- `longcat_video_avatar_comfyui_gguf/pytorch/loader.py` (partial fix, in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 50a2163941 |
