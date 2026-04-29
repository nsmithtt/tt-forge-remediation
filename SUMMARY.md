# Remediation Summary: flexan_kshitijthakkar_qwen3_5_moe_0_87b_d0_8b_gguf-causal_lm-pytorch-MoE_0.87B_d0.8B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flexan_kshitijthakkar_qwen3_5_moe_0_87b_d0_8b_gguf/causal_lm/pytorch-MoE_0.87B_d0.8B_GGUF-single_device-inference]

## Result
FAIL — loader bugs fixed (GGUF chain, SSM config, conv1d shape, expert tensors), but test crashes with segfault in tt-xla TorchFunctionMode during Dynamo CPU-fallback graph partitioning

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
tt-xla-segfault-torch-function-mode-cpu-fallback

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (before any fixes):
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
This is a swig DeprecationWarning shown in pytest's warnings summary, not the actual error.

Actual original failure:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
```
With a LOAD REPORT showing multiple weight mismatches:
- `linear_attn.out_proj.weight`: ckpt [1024, 2048] vs model [1024, 4096]
- `linear_attn.conv1d.weight`: ckpt [6144, 4] vs model [8192, 1, 4]
- `mlp.experts.down_proj`: ckpt [8, 1024, 400] vs model [8, 1024, 512]
- `mlp.shared_expert.gate_proj.weight`: ckpt [400, 1024] vs model [512, 1024]
- `linear_attn.A_log`: ckpt [16] vs model [32]
- `linear_attn.in_proj_qkv.weight`: ckpt [6144, 1024] vs model [8192, 1024]

After all loader fixes, the model loaded without mismatches and reached TT hardware. The test then crashed with:
```
Fatal Python error: Segmentation fault
```
in `tt_torch/torch_overrides.py:34` (`TorchFunctionOverride.__torch_function__` → `return func(*args, **kwargs)`)
while executing `partition_fx_graph_for_cpu_fallback` during TorchDynamo graph compilation of `Qwen3_5MoeForCausalLM.forward` at line 1487.

## Root cause
**Loader layer (fixed):** The Flexan kshitijthakkar-qwen3.5-moe-0.87B-d0.8B-GGUF model uses the `qwen35moe` GGUF architecture string for a 24-layer SSM/Mamba-attention hybrid (every 4th layer is full attention, others use GatedDeltaNet linear attention) with MoE FFN (8 experts, top-2, expert FFN size 400). Four loader bugs combined:

1. **Broken GGUF loader chain** (`gguf-load-checkpoint-model-to-load-kwarg`): The original loader used import-time monkey-patching of `load_gguf_checkpoint`, joining a chain of 26 broken patchers that drop the `model_to_load` kwarg added in transformers 5.x. Fixed by replacing with a context-manager that traces the chain to the real transformers function.

2. **Missing GGUF config field mappings**: `Qwen3_5MoeTextConfig` defaults (`moe_intermediate_size=512`, `linear_num_value_heads=32`, etc.) did not match GGUF metadata (expert FFN size=400, ssm.time_step_rank=16). Fixed by extending `GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"]` with SSM parameters (`ssm.time_step_rank`, `ssm.state_size`, `ssm.conv_kernel`) and MoE feed-forward sizes (`expert_feed_forward_length`, `expert_shared_feed_forward_length`), plus mirroring symmetric parameters (`linear_num_value_heads` from key heads).

3. **ssm_conv1d.weight shape mismatch**: GGUF stores `ssm_conv1d.weight` as `[C, kernel]` but HF grouped `Conv1d` expects `[C, 1, kernel]`. Fixed by creating `Qwen35MoeTensorProcessor` (subclass of `Qwen2MoeTensorProcessor`) with `np.expand_dims(weights, axis=1)` when `"ssm_conv1d.weight" in name`.

4. **Split expert tensor key aliasing**: GGUF stores expert weights separately as `ffn_gate_exps` and `ffn_up_exps` but the arch name_map generates `ffn_gate_up_exps` (merged form). Fixed by injecting both split-tensor GGUF keys in `patched_get_gguf_hf_weights_map`.

**Compiler-stack layer (unfixed):** After the loader fixes, the model loaded cleanly (441 GGUF tensors loaded without mismatches) and entered TT hardware compilation. After ~17 minutes of compilation (3 graph segments), the process crashed with SIGSEGV at `tt_torch/torch_overrides.py:34`. This is the `TorchFunctionOverride` global `TorchFunctionMode` (entered at module import via `torch_function_override.__enter__()`). The crash occurs inside `partition_fx_graph_for_cpu_fallback` in `torch_xla/_dynamo/dynamo_bridge.py:762`, which runs graph nodes to determine CPU-fallback partitioning. The globally-active `TorchFunctionOverride` intercepts these calls and the underlying `func(*args, **kwargs)` call crashes with SIGSEGV — likely because the tensors involved are in an XLA device state incompatible with the CPU-fallback execution context.

## Fix
**Loader fixes (complete)** — in `tt_forge_models` on branch `remediation/flexan_kshitijthakkar_qwen3_5_moe_0_87b_d0_8b_gguf-causal_lm-pytorch-MoE_0.87B_d0.8B_GGUF-single_device-inference`:
- New file: `flexan_kshitijthakkar_qwen3_5_moe_0_87b_d0_8b_gguf/causal_lm/pytorch/loader.py` (complete rewrite with context-manager patching, `Qwen35MoeTensorProcessor`, full SSM+MoE config mappings)
- Modified: `**kwargs` passthrough in 26 other GGUF loaders' `_patched_load_gguf_checkpoint` wrappers

**Compiler-stack fix (proposed)** — in `tt-xla/python_package/tt_torch/torch_overrides.py`:
The `TorchFunctionOverride` is entered globally at import time. During `partition_fx_graph_for_cpu_fallback`, torch_xla runs FX graph nodes on the CPU to determine which ops fall back to CPU. The global `TorchFunctionMode` intercepts these calls and passes tensors that may be in an XLA state to the underlying function, causing SIGSEGV. The fix would be to guard `TorchFunctionOverride.__torch_function__` against activation during graph partitioning (e.g., by detecting `torch.compiler.is_compiling()` or the XLA graph-extraction context), or to temporarily disable the TorchFunctionMode during `partition_fx_graph_for_cpu_fallback`. This requires understanding whether the SIGSEGV is from a null XLA tensor reference, a C extension data-type mismatch, or another cause — which needs a debugger/core dump.

## Tier B justification
`internal-error-unknown-mechanism` — the crash is a SIGSEGV inside a C extension (`torch._C`). The exact op and tensor state that triggers the fault are unknown without a core dump or debugger session. Diagnosing the root cause (null XLA pointer, wrong device tensor, type confusion) must precede any code change.

## Verification
- pytest exit: FAIL (SIGSEGV, process killed by signal)
- Hardware:    blackhole-p150b
- Duration:    ~22min 38s (model load: ~4min 46s, MLIR compilation: ~17min 52s, then crash)
- Tier A attempts: N/A

## Files changed
**tt_forge_models (remediation branch):**
- `flexan_kshitijthakkar_qwen3_5_moe_0_87b_d0_8b_gguf/causal_lm/pytorch/loader.py` (new)
- 26 other GGUF loaders (kwargs fix in `_patched_load_gguf_checkpoint`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2b9081907855aa4317a4b7217b92105344c8503e |
| tt-forge-models | b6d991254a26598640a8ef6a55b264d031388d7f |
