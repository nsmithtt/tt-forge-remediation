# Remediation Summary: ernie_4_5_21b_a3b_pt_gguf-causal_lm-pytorch-ERNIE_4_5_21B_A3B_PT_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ernie_4_5_21b_a3b_pt_gguf/causal_lm/pytorch-ERNIE_4_5_21B_A3B_PT_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — segfault in partition_fx_graph_for_cpu_fallback during MoE expert graph partitioning (Tier B)

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
grouped-mm-cpu-fallback-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original error:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Terminal failure after loader fixes:
```
Fatal Python error: Segmentation fault

Current thread 0x0000786c9a24d140 (most recent call first):
  File ".../torch/_ops.py", line 841 in __call__
  File ".../tt_torch/torch_overrides.py", line 39 in __torch_function__
  File ".../torch/_ops.py", line 841 in __call__
  File ".../torch/fx/interpreter.py", line 336 in call_function
  File ".../torch/fx/interpreter.py", line 256 in run_node
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File ".../torch/fx/interpreter.py", line 174 in run
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
```

## Root cause

Five loader bugs were fixed before reaching the terminal failure:

1. **Missing requirements.txt** (`gguf>=0.10.0` absent) caused the original ImportError.
2. **GGUF architecture unregistered**: `ernie4_5-moe` not in transformers 5.x GGUF tables; added `_patch_transformers_ernie4_5_moe_gguf()` to register it in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `TENSOR_PROCESSORS`, and `GGUF_TO_FAST_CONVERTERS`, plus patching `load_gguf_checkpoint` to remap `model_type` from `ernie4_5-moe` to `ernie4_5_moe` for AutoConfig, and `get_gguf_hf_weights_map` for gguf-py lookup.
3. **26 GGUF loaders with fixed-signature patcher**: transformers 5.x `load_gguf_checkpoint` added a `model_to_load=` kwarg; all 26 loaders using a fixed-signature `_patched_load_gguf_checkpoint` rejected it; fixed with `*args, **kwargs` forwarding.
4. **`get_xla_device_arch()` abort**: calling `xr.global_runtime_device_attributes()` before the PJRT client is initialized causes `GetComputationClientOrDie()` to SIGABRT; added guard `if not torch_xla._XLAC._xla_computation_cache_is_initialized(): return ""` in `tt-xla/tests/runner/test_utils.py`.
5. **histc on integer tensors**: ERNIE 4.5 MoE's router calls `torch.histc` on Long expert-index tensors; `histogram_cpu` is not implemented for integer dtypes; fixed in `TorchFunctionOverride.__torch_function__` with a substring check (`"histc" in getattr(func, "__name__", "")`) and upcast to float.

The terminal failure is in `tt-xla`: during `partition_fx_graph_for_cpu_fallback`, the FX interpreter calls a grouped-matmul op (MoE expert dispatch) on CPU. The op segfaults at `torch._ops.py:841` dispatching through `TorchFunctionOverride`. This is the same `grouped-mm-cpu-fallback-segfault` seen in Qwen3MoE, GLM-4.7 MoE Lite, and LFM2-MoE models — the CPU fallback partitioner crashes on grouped expert matmul operations.

## Fix

Loader fixes (tt_forge_models remediation branch):
- `third_party/tt_forge_models/ernie_4_5_21b_a3b_pt_gguf/causal_lm/pytorch/requirements.txt` — created with `gguf>=0.10.0`
- `third_party/tt_forge_models/ernie_4_5_21b_a3b_pt_gguf/causal_lm/pytorch/loader.py` — added `_patch_transformers_ernie4_5_moe_gguf()` at import time; added `chat_template` guard in `load_inputs`
- 26 GGUF loader files — changed `_patched_load_gguf_checkpoint` from fixed-signature to `*args, **kwargs` forwarding

Frontend fixes (tt-xla remediation branch):
- `python_package/tt_torch/torch_overrides.py` — histc int→float cast via substring check in `TorchFunctionOverride.__torch_function__`
- `tests/runner/test_utils.py` — guard `get_xla_device_arch()` against uninitialized PJRT client

Proposed fix for terminal bug (not attempted — Tier B):
- The `partition_fx_graph_for_cpu_fallback` function in `torch_xla/_dynamo/dynamo_bridge.py` needs to handle grouped expert matmul ops without segfaulting. The fix would require either: (a) registering a CPU fallback kernel for the grouped-mm op that MoE models use, or (b) preventing those ops from being routed to the CPU fallback partitioner entirely. Either approach is cross-cutting and requires new infrastructure in the XLA Dynamo bridge.

## Tier B justification
**cross-cutting**: The segfault in `partition_fx_graph_for_cpu_fallback` (dynamo_bridge.py) occurs because the CPU fallback partitioner attempts to execute MoE grouped-matmul ops that have no safe CPU kernel path. Fixing this requires either registering CPU fallback kernels for these ops or restructuring how the partitioner handles them — both of which involve changes across the Dynamo bridge and the TT backend op registry.

## Verification
- pytest exit: FAIL (segmentation fault — process crashed)
- Hardware:    blackhole-p150b
- Duration:    ~19 minutes (18:38–18:57 UTC)
- Tier A attempts: N/A

## Files changed
**tt_forge_models** (remediation branch `remediation/ernie-4-5-21b-a3b-pt-gguf-causal-lm-pytorch-ernie-4-5-21b-a3b-pt-q4-k-m-gguf-single-device-inference`):
- `ernie_4_5_21b_a3b_pt_gguf/causal_lm/pytorch/requirements.txt` (created)
- `ernie_4_5_21b_a3b_pt_gguf/causal_lm/pytorch/loader.py` (GGUF arch patch + chat_template guard)
- 26 GGUF loader files (kwargs forwarding fix)

**tt-xla** (remediation branch `remediation/ernie-4-5-21b-a3b-pt-gguf-causal-lm-pytorch-ernie-4-5-21b-a3b-pt-q4-k-m-gguf-single-device-inference`):
- `python_package/tt_torch/torch_overrides.py` (histc int→float cast)
- `tests/runner/test_utils.py` (PJRT init guard)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a76dd22aa78fab88804b6e74fdbb494f67a7b1e3 |
| tt-forge-models | 75e722f2cc1f36f5774fff13deb1105d6130f7db |
