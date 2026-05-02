# Remediation Summary: qwen_3_5_27b_ultimate_irrefusable_heretic_i1_gguf-causal_lm-pytorch-ultimate_irrefusable_heretic_i1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_27b_ultimate_irrefusable_heretic_i1_gguf/causal_lm/pytorch-ultimate_irrefusable_heretic_i1_Q4_K_M-single_device-inference]

## Result
XFAIL — 27B BF16 model (~54 GB) exceeds single-device DRAM (32 GB p150b)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-bf16-exceeds-32gb-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (runs 1–2):
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
```
Cause: wrong model class (`Qwen3ForCausalLM` instead of `Qwen3_5TextForCausalLM`) because
`load_gguf_checkpoint` was contaminated by another loader that remaps `qwen35→qwen3`.

Terminal failure (run 3, after loader fixes):
```
TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 178257920 B DRAM buffer across 8 banks,
where each bank needs to store 22282240 B, but bank size is 4273390016 B
(allocated: 4029953344 B, free: 243436672 B, largest free block: 14417920 B)
```

## Root cause
Two sequential issues:

**Issue 1 — loader (fixed)**: The loader had no GGUF architecture registration for `qwen35`. This
caused two problems:
1. `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` was patched at module-import
   time by `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/loader.py` (and 25 other loaders) with a
   narrow-signature wrapper `(gguf_path, return_tensors=False)`. transformers 5.2.0 added a
   `model_to_load` keyword argument, causing `TypeError` at model load time.
2. Even after widening all narrow-sig patches to `(*args, **kwargs)`, the tvall43 loader's
   `_patched_load_gguf_checkpoint` maps `qwen35→qwen3`, so the target model was instantiated as
   `Qwen3ForCausalLM` (q_proj = `num_heads * head_dim`) instead of `Qwen3_5TextForCausalLM`
   (q_proj = `num_heads * head_dim * 2`). This produced shape mismatch `[6144, 5120]` vs
   `[12288, 5120]` for all full-attention q_proj weights.

   The fix adds a complete `_patch_transformers_qwen35_gguf()` that:
   - Registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`
   - Maps all qwen35 GGUF metadata keys to `Qwen3_5TextConfig` fields, critically mapping
     `ssm.time_step_rank→linear_num_value_heads` (value = 48 for 27B, vs default 32)
   - Installs `Qwen35TensorProcessor` for `dt_bias→dt_proj` rename, ssm_conv1d shape fix,
     and `ssm_a = log(-ssm_a)` transform
   - Translates `qwen35→qwen3_5_text` in `load_gguf_checkpoint`, also building `layer_types`
     from `full_attention_interval=4` (every 4th layer is full_attention)
   - Patches `get_gguf_hf_weights_map` to reverse `qwen3_5_text→qwen35` for weight lookup
   - Detects contamination: when another loader has already remapped `qwen35→qwen3`, checks for
     `linear_num_value_heads` in config dict (an SSM-specific field only present after the
     qwen35 arch config mapping fires) to still catch and re-remap to `qwen3_5_text`

**Issue 2 — hardware-class (XFAIL)**: The 27B model has ~27B parameters. At BFloat16
(transformers dequantizes GGUF Q4_K_M to BF16 before sending to device) this is ~54 GB.
The p150b BlackHole card has 32 GB DRAM. The allocator reached 30 GB allocated and then
failed trying to allocate a 170 MB buffer. The model cannot fit on a single device.

## Fix
**Loader fix** (`loader` layer):
- `tt-xla/third_party/tt_forge_models/qwen_3_5_27b_ultimate_irrefusable_heretic_i1_gguf/causal_lm/pytorch/loader.py`:
  Full rewrite adding `_patch_transformers_qwen35_gguf()` with qwen35 arch registration,
  SSM metadata mapping, tensor processor, `load_gguf_checkpoint` patch (with contamination
  detection), and `get_gguf_hf_weights_map` patch. `load_inputs` adds `use_cache=False` to
  prevent `Qwen3_5DynamicCache` TypeError in the inference evaluator.
- 26 other GGUF loaders: widened `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
  to `(*args, **kwargs)` to accept the `model_to_load` kwarg added in transformers 5.2.0.

**Hardware-class disposition** (`hardware-class`):
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `KNOWN_FAILURE_XFAIL` entry for this test with OOM message and capacity explanation.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL (OOM at device execution, after loader fixes resolved all prior errors)
- Hardware:    blackhole-p150b
- Duration:    776.57s (run 2, loader mismatch) / 11981s (run 3, OOM at execution after ~3h compilation)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/qwen_3_5_27b_ultimate_irrefusable_heretic_i1_gguf/causal_lm/pytorch/loader.py` (full rewrite with qwen35 arch registration and contamination detection)
- `tt-xla/third_party/tt_forge_models/*/causal_lm/pytorch/loader.py` ×26 (narrow-sig patch widened to `*args, **kwargs`)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 25cf3b9b44d514c61a1bb9c9a9667bd16b28a317 |
| tt-forge-models | d379f52659fce2cc9f865618176cd6784edb104b |
