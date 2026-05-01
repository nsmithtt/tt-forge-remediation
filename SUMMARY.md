# Remediation Summary: mradermacher_qwen3_5_2b_claude_4_6_opus_reasoning_distilled_heretic_v0_i1_gguf-causal_lm-pytorch-2B_Claude_4.6_Opus_Reasoning_Distilled_heretic_v0_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_2b_claude_4_6_opus_reasoning_distilled_heretic_v0_i1_gguf/causal_lm/pytorch-2B_Claude_4.6_Opus_Reasoning_Distilled_heretic_v0_i1_GGUF-single_device-inference]

## Result
FAIL — qwen3.5 hybrid GLA+full-attention architecture not supported by transformers qwen3 class (Tier B new-infrastructure)

## Stack layer
loader, new-infrastructure

## Tier
B

## Bug fingerprint
qwen35-hybrid-full-attn-layer-size-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
First failure (fixed): `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

Terminal failure:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!

model.layers.{3, 7, 11, 15, 19, 23}.self_attn.k_norm.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([256]) vs model:torch.Size([128])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.q_proj.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([4096, 2048]) vs model:torch.Size([1024, 2048])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.q_norm.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([256]) vs model:torch.Size([128])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.v_proj.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([512, 2048]) vs model:torch.Size([256, 2048])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.o_proj.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([2048, 2048]) vs model:torch.Size([2048, 1024])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.k_proj.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([512, 2048]) vs model:torch.Size([256, 2048])
```

## Root cause

**Fixed (loader):** 26 qwen3.5 GGUF loaders had a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` narrow signature that does not accept the `model_to_load` keyword argument added in transformers 5.2.0. During test collection all loaders are imported, installing the narrow-signature patch globally. When `from_pretrained` later called `load_gguf_checkpoint(..., model_to_load=dummy_model)`, it hit the narrow-signature patch and raised `TypeError`. Additionally, this loader (`mradermacher_qwen3_5_2b_claude`) had no patch of its own, so it relied on and was polluted by other loaders' patches.

**Terminal (Tier B):** The Qwen3.5 2B model uses a hybrid GLA+full-attention architecture with `full_attention_interval=4`. Layers 3, 7, 11, 15, 19, 23 (every 4th starting at 3) are full-attention layers with `num_heads=32, num_kv_heads=4` (q_proj=[4096,2048], k_proj=[512,2048]), while the remaining layers use GLA with `num_heads=8, num_kv_heads=2` (q_proj=[1024,2048], k_proj=[256,2048]). The transformers qwen3 class has no knowledge of this per-layer head-count variation and applies the GLA configuration to all layers, causing 6 layers to have mismatched attention projection shapes when the GGUF checkpoint is loaded.

## Fix

**Loader fix (committed):** Updated all 26 narrow-signature loaders from `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `def _patched_load_gguf_checkpoint(*args, **kwargs)` and updated the internal call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Also added the full qwen35 arch support patch to the `mradermacher_qwen3_5_2b_claude` loader which previously had no patch at all.

**Terminal fix (proposed, Tier B):** A proper transformers model class for qwen3.5 hybrid (or a custom GGUF weight loading procedure that maps per-layer head counts from GGUF metadata to the correct layer configs) would be required. This needs new infrastructure in the transformers library or a custom model class in tt-forge-models with per-layer config support.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

Loading the qwen3.5 hybrid GGUF correctly requires either a new transformers model class that supports per-layer attention head counts (GLA dims vs full-attention dims), or a custom weight conversion that restructures the GGUF tensors to fit the qwen3 class. Neither can be done as a scoped single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    292.93s (second run after loader fix)
- Tier A attempts: N/A

## Files changed
- `mradermacher_qwen3_5_2b_claude_4_6_opus_reasoning_distilled_heretic_v0_i1_gguf/causal_lm/pytorch/loader.py` — added qwen35 arch support patch with `*args, **kwargs`
- 26 other qwen3.5 GGUF loaders — updated `_patched_load_gguf_checkpoint` signature from narrow to `*args, **kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3b9b4336761eee88c39be2363f2bb7509b134231 |
| tt-forge-models | 1b29944c45795582955c0189a10c92edbaf3ab33 |
