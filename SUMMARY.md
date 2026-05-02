# Remediation Summary: nexveridian_qwen3_5_35b_a3b_3bit-causal_lm-pytorch-35B_A3B_3bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nexveridian_qwen3_5_35b_a3b_3bit/causal_lm/pytorch-35B_A3B_3bit-single_device-inference]

## Result
FAIL — Tier B SIGSEGV in partition_fx_graph_for_cpu_fallback during Dynamo compilation of Qwen3.5-MoE with GDA (GatedDeltaNet) layers

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
dynamo-bridge-partition-fx-sigsegv

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python crash in `partition_fx_graph_for_cpu_fallback` (`dynamo_bridge.py:762`) during torch.compile of `Qwen3_5MoeForCausalLM`. The GDA `linear_attn` layers (30/40 layers) cause a graph break at `modeling_qwen3_5_moe.py:1487`; when Dynamo attempts to split the FX graph for CPU fallback it SIGSEGVs inside `torch_overrides.py:34 __torch_function__`.

Stack (innermost first):
```
torch/_ops.py:841 __call__
tt_torch/torch_overrides.py:34 __torch_function__
torch/fx/interpreter.py:336 call_function
torch_xla/_dynamo/dynamo_bridge.py:652 run_node
torch_xla/_dynamo/dynamo_bridge.py:762 partition_fx_graph_for_cpu_fallback
torch_xla/_dynamo/dynamo_bridge.py:859 extract_compiled_graph_helper
tt_torch/backend/backend.py:215 _call_experimental_compile
```

## Root cause
The `Qwen3_5MoeDecoderLayer` alternates between full-attention (`self_attn`) and GatedDeltaNet linear-attention (`linear_attn`) layers. The GDA layers contain `conv1d` and recurrent state-update operations that Dynamo cannot trace without a graph break. When `partition_fx_graph_for_cpu_fallback` in `torch_xla/dynamo_bridge.py` runs the FX graph interpreter to probe which ops must fall back to CPU, calling into `tt_torch/torch_overrides.py:__torch_function__` SIGSEGVs. This is the same class of bug as seen in the Qwen3.5-27B MLX NVFP4 and Qwen3.5-35B-A3B GGUF variants — the XLA bridge cannot safely partition graphs containing GDA recurrence ops.

## Fix
Five loader bugs were fixed in `nexveridian_qwen3_5_35b_a3b_3bit/causal_lm/pytorch/loader.py` in `tenstorrent/tt-forge-models`:

1. **load_shard_spec hasattr guard + use_cache=False** (`6a01fc0939`): Added `hasattr(layer, "self_attn")` / `elif hasattr(layer, "linear_attn")` guards in `load_shard_spec` for the hybrid decoder layer, and added `inputs["use_cache"] = False` in `load_inputs` to prevent `Qwen3_5MoeDynamicCache` (not a `Cache` subclass) from breaking the test evaluator's `tree_map`.

2. **Custom safetensors loader with MLX 3-bit dequantization** (`89fd23fa1b`): Replaced `AutoModelForCausalLM.from_pretrained` with a custom loader that reads the safetensors shards directly, strips the `language_model.` VLM container prefix from all 1757 weight keys, and dequantizes MLX affine group-quantized `uint32` weights (3-bit cross-boundary packing, per-group bfloat16 scales+biases).

3. **Per-tensor quantization bit-width overrides** (`17a3fba753`): The quantization config has per-tensor overrides for `mlp.gate` and `mlp.shared_expert_gate` (bits=8 instead of default bits=3). Added lookup of per-tensor `bits` and `group_size` from `full_config.quantization` so gate tensors dequantize correctly.

4. **N-dim expert tensor support in dequantize** (`406f8b4dc9`): MoE expert weights `switch_mlp.{gate,up,down}_proj.weight` are stored as 3D tensors `[num_experts, out_features, packed_in]`. Generalized `_dequantize_mlx_affine` to handle arbitrary leading dims by flattening to 2D before unpacking and reshaping back.

5. **conv1d MLX→PyTorch layout permute** (`94ff25fe64`): MLX stores Conv1d weights as `(C, K, 1)`; PyTorch expects `(C, 1, K)`. Added `.permute(0, 2, 1).contiguous()` for `linear_attn.conv1d.weight` during state dict load.

6. **Split expert weight remapping to batched layout** (`0725da1070`): The checkpoint stores MoE experts as separate `switch_mlp.gate_proj.weight` and `switch_mlp.up_proj.weight` tensors `[256, 512, hidden]`; transformers `Qwen3_5MoeExperts` expects a single batched `experts.gate_up_proj` of shape `[256, 1024, hidden]` (gate and up concatenated on dim=1). Added post-processing to remap and merge these tensors, and rename `switch_mlp.down_proj.weight` → `experts.down_proj`.

The terminal SIGSEGV is in tt-xla and cannot be fixed at the loader level.

Proposed fix for the Tier B bug: In `torch_xla/_dynamo/dynamo_bridge.py:partition_fx_graph_for_cpu_fallback`, the FX graph interpreter calls `torch_overrides.__torch_function__` with a fake tensor on a graph break boundary, which crashes. The fix would involve either (a) guarding the interpreter from calling TT-specific overrides during the partitioning probe phase, or (b) preventing graph breaks from occurring in the GDA path by annotating GDA layers with `@torch.compiler.disable` — but the latter is forbidden by remediation rules. This requires tt-xla expertise and is a cross-cutting change.

## Tier B justification
Which indicator: `internal-error-unknown-mechanism` / `cross-cutting`

The SIGSEGV occurs inside C-level code reachable from `__torch_function__` in the XLA bridge during FX graph partitioning. There is no Python-level error string captured — the process exits with a crash dump. Diagnosing the exact crash site requires a native debugger attached to the process. Fixing it would require changes to `dynamo_bridge.py` and/or `torch_overrides.py` to guard against this call path, which is a cross-cutting change affecting all models that trigger graph breaks.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 1722.15s (0:28:42) — crash during compilation phase
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: nexveridian_qwen3_5_35b_a3b_3bit/causal_lm/pytorch/loader.py` (6 commits on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8481f8799aa9455ef4884ad80460b384df2a3b91 |
| tt-forge-models | 0725da10704c0dd93c7ed6330b43be0d8b3422e3 |
