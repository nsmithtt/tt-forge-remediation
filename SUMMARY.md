# Remediation Summary: abhiray_huihui_qwen3_5_9b_abliterated_gguf-causal_lm-pytorch-9B_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch-9B_Abliterated_GGUF-single_device-inference]

## Result
FAIL — PCC=0.8600862920171023 (required 0.99); root cause is dynamic_update_slice ops from torch_chunk_gated_delta_rule being CPU-hoisted by tt-mlir, not run natively on device

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
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8600862920171023. Required: pcc=0.99.
```

## Root cause
Qwen3.5-9B uses a hybrid GatedDeltaNet (linear-attention SSM) + full-attention architecture. The SSM recurrence in each GatedDeltaNet layer is implemented by `torch_chunk_gated_delta_rule` (the pure-Python fallback; the `fla` flash-linear-attention package is not installed). This fallback contains Python for-loops with in-place variable-index tensor updates:

```python
attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
```

These compile to `dynamic_update_slice` StableHLO operations. tt-mlir does not lower `dynamic_update_slice` to device-native ops; instead the `cpu-hoist-non-lowerable-shlo-ops` pass hoists them to CPU. With 24 GatedDeltaNet layers each containing a chunk-size inner loop (chunk_size=16, 8 chunks for seq_len=128), every forward pass incurs many CPU↔TT device transfers. The accumulated precision differences across all these transfers and the surrounding float32 → bfloat16 type conversions produce PCC=0.86, well below the required 0.99.

All four loader-layer bugs were fixed prior to this PCC failure (see Fix section). The model loads, runs on both CPU and TT silicon, and produces correct output shapes — the PCC gap is purely from the missing device-native `dynamic_update_slice` lowering.

## Fix
Four loader fixes were committed to tt_forge_models on branch `remediation/abhiray_huihui_qwen3_5_9b_abliterated_gguf-causal_lm-pytorch-9B_Abliterated_GGUF-single_device-inference`:

1. **`1a9e889150`** — Add qwen35 GGUF architecture support: registers "qwen35" in `GGUF_SUPPORTED_ARCHITECTURES`, adds the qwen35→HF config-field mapping (including `full_attention_interval`), and wires up the qwen3 fast-tokenizer converter.  
   File: `abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`

2. **`d72b4d6fd7`** — Load as `qwen3_5_text` not `qwen3`: adds `_get_raw_gguf_arch()` to read `general.architecture` directly from the GGUF header (bypassing the mradermacher loader's patch chain that renames qwen35→qwen3), and sets `result["config"]["model_type"] = "qwen3_5_text"` so `Qwen3_5ForCausalLM` is instantiated instead of `Qwen3ForCausalLM`.  
   File: `abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`

3. **`6767f366b5`** — Add `_Qwen35TensorProcessor`: `perform_fallback_tensor_mapping` maps `blk.N.ssm_dt.bias` → `linear_attn.dt_bias` (gguf-py incorrectly maps `blk.N.ssm_dt` → a non-existent `dt_proj`); `process` unsqueezes `ssm_conv1d.weight` from `[out_ch, kern]` to `[out_ch, 1, kern]` to match `nn.Conv1d` expectations.  
   File: `abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`

4. **`dfd5901933`** — Pass `use_cache=False` in `load_inputs()`: `Qwen3_5DynamicCache` is not a PyTree-registered type; without this, the test infrastructure's `torch.equal()` comparison on raw model output fails with `TypeError: equal() argument must be Tensor, not Qwen3_5DynamicCache`.  
   File: `abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py`

The remaining PCC failure requires device-native `dynamic_update_slice` lowering in tt-mlir. The proposed fix would be implementing a lowering pattern from `stablehlo.dynamic_update_slice` to a TTIR/TTNN equivalent (e.g. using slice/concat or a scatter primitive) in `lib/Conversion/StableHLOToTTIR/`. This is Tier B: it requires new lowering infrastructure touching multiple passes and possibly new TTIR ops.

## Tier B justification
**new-infrastructure**: Device-native `dynamic_update_slice` lowering does not exist in tt-mlir. Implementing it requires adding a new conversion pattern in `StableHLOToTTIR`, a corresponding TTIR op definition, and a TTIR→TTNN lowering — cross-cutting changes spanning at minimum 3 files, likely requiring new op infrastructure.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 1952.16s (0:32:32)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/abhiray_huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch/loader.py` (4 commits)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 93cbf14c2ded9ccdf47a14bbe20246583062d57a |
| tt-forge-models | dfd5901933 |
