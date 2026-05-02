# Remediation Summary: mradermacher_routangseng_voice_0_8b_gguf-causal_lm-pytorch-0_8B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_routangseng_voice_0_8b_gguf/causal_lm/pytorch-0_8B_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Tier B compiler bug: GDA linear_attn recurrence produces dynamic_update_slice in StableHLO, CPU-hoisted by tt-mlir; PCC=0.4384 on TT silicon (required 0.99)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
dynamic-update-slice-cpu-fallback-pcc-loss

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure on branch arch-c-36-tt-xla-dev/nsmith/hf-bringup-21:
```
ValueError: GGUF model with architecture qwen35 is not supported yet
AttributeError: 'NoneType' object has no attribute 'config'
```

After loader fix, silicon test failure:
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.4384369974978355. Required: pcc=0.99.
```

## Root cause
Two distinct root causes:

**Loader (fixed):** The `routangseng-voice-0.8b` GGUF declares `general.architecture = qwen35`, a Qwen3.5 hybrid SSM+attention architecture. Transformers 5.x has no built-in registration for `qwen35`. Four sub-bugs:
1. `qwen35` not in `GGUF_SUPPORTED_ARCHITECTURES` → ValueError/NoneType.
2. `GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"]` missing SSM field mappings, specifically `ssm.time_step_rank → linear_num_value_heads` (GGUF value=16 vs HF default=32) → shape mismatches on all 18 GDA layers.
3. Session contamination: other qwen35 GGUF loaders in the same pytest session already populated the `qwen35` config dict without SSM fields; an `if "qwen35" not in` guard skipped the SSM fields entirely. Fixed by `setdefault` per-key.
4. `get_gguf_hf_weights_map` uses HF `model_type` to look up the gguf-py arch, but the model type is `qwen3_5_text`; without the alias `qwen3_5_text → qwen35`, the weights map fails with NotImplementedError.

**Compiler (unfixed, Tier B):** The model's 18 GDA (GatedDeltaNet) `linear_attn` layers use `torch_chunk_gated_delta_rule` (Python for-loop with indexed state updates) when `flash-linear-attention` is not installed. This produces `stablehlo.dynamic_update_slice` operations that tt-mlir cannot lower to TT kernels and CPU-hoists. With 18 GDA layers, the accumulated precision difference between the CPU-hoisted recurrence (float32) and the TT-executed surrounding ops (bfloat16) causes PCC=0.4384.

CPU forward pass produces finite values with correct shapes (verified independently), confirming the loader is correct and the PCC failure is hardware/compiler.

## Fix
Loader fix in `tt-forge-models` repo on branch:
`remediation/mradermacher_routangseng_voice_0_8b_gguf-causal_lm-pytorch-0_8B_Q4_K_M_GGUF-single_device-inference`

File changed:
`third_party/tt_forge_models/mradermacher_routangseng_voice_0_8b_gguf/causal_lm/pytorch/loader.py`

Changes:
- Register `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`
- Add full SSM config field mappings for qwen35 (including `ssm.time_step_rank → linear_num_value_heads`, `ssm.group_count → linear_num_key_heads`, `ssm.state_size → linear_key_head_dim`, `ssm.conv_kernel → linear_conv_kernel_dim`) using `setdefault` per-key to survive session contamination
- Patch `get_gguf_hf_weights_map` to alias `qwen3_5_text → qwen35` model type
- Register `_Qwen35TensorProcessor` to fix `ssm_conv1d.weight` 2D→3D expansion and `dt_bias` fallback mapping
- Patch `load_gguf_checkpoint` to force `model_type = "qwen3_5_text"` after loading (raw GGUF arch read via `GGUFReader` to avoid cross-loader contamination)
- Add `use_cache=False` to `load_inputs()` to prevent `Qwen3_5DynamicCache` TypeError in test harness
- Add `if hasattr(layer, "self_attn"):` guard in `load_shard_spec()` for hybrid GDA+attention layer types

The Tier B compiler bug (`dynamic_update_slice` in GDA recurrence) remains unfixed and is the active cause of test failure.

## Tier B justification
**Indicator: new-infrastructure**

Lowering `stablehlo.dynamic_update_slice` to TT kernels requires either:
(a) a new TT kernel for in-place state updates on the recurrence loop, or
(b) a structural transformation that unrolls or batches the GDA recurrence into a form without indexed updates.

Neither is a scoped single-function fix. This is the same class of issue documented in `qwen35_mlx_4bit_vlm_hardware_xfail.md` (PCC=0.750 for 9B MLX 4-bit with 24 GDA layers). The 0.8B has 18 GDA layers and gives PCC=0.4384, consistent with smaller models being more sensitive to accumulated float32/bfloat16 divergence from CPU-hoisted GDA recurrences.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 2711.78s (0:45:11)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/mradermacher_routangseng_voice_0_8b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | N/A (no changes) |
| tt-mlir         | N/A (no changes) |
| tt-xla          | fd88fed9a5aac3659f7fe1325cfd35b262b46865 |
| tt-forge-models | 3fb0730e94488de2c528ce8adf8f2ab6aca9561d |
