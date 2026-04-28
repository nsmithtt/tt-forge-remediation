# Remediation Summary: c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch-c4ai_command_r7b_12_2024_abliterated_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch-c4ai_command_r7b_12_2024_abliterated_GGUF-single_device-inference]

## Result
FAIL — second compiler-stack bug: TT device PCC=0.9874 vs CPU bfloat16 baseline; gap (0.012) exceeds what bf16 format alone explains (CPU f32 vs CPU bf16 = 0.9997), indicating hardware bfloat16 accumulation differences not accounted for by the consteval-on-host pattern; first Tier A fix (aten.slice.Tensor bounds) was committed

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: YES — measured BF16-CPU vs FP32-CPU PCC = 0.9997; TT vs CPU (both bf16) = 0.9874; gap of 0.012 beyond the 0.0003 bf16/f32 delta is not purely format-level accumulation; threshold NOT lowered per skill rules
- Warning / exception suppression: NO

## Failure
Original reported failure (2026-04-25 CI):
```
2026-04-25 03:08:42.044 | critical |          Always | TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology. Inter-mesh mapping failed after 2 attempt(s). Logical meshes being mapped: [0] (1 total). Physical meshes available: [0] (1 total). Failed mesh pair configurations tried: 1 out of 1 possible combinations. Inter-mesh validation mode: STRICT. Solver error: Mapping validation failed: 1 target node(s) are not mapped to any global node: 0. Failed mesh pairs from previous attempts: [(logical=0, physical=0)].. Either relax pinnings or modify the MGD. If this is unexpected, run ./build/test/tt_metal/tt_fabric/test_system_health to check connectivity. (assert.hpp:104)
```

Reproduction failure 1 — current submodule HEAD (loader bug, cohere2 not registered):
```
ValueError: GGUF model with architecture cohere2 is not supported yet.
```

Reproduction failure 2 — after loader fixes, before Tier A fix (compiler bug):
```
RuntimeError: Value out of range (expected to be in range of [-74, 73], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_3, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  transformers/cache_utils.py:214, in DynamicSlidingWindowLayer.update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

Post-Tier-A-fix failure (second compiler-stack bug, PCC):
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9874482860896393. Required: pcc=0.99.
```

## Root cause

**Loader bug 1 (fixed):** The Cohere2 GGUF model uses architecture tag `cohere2`. Transformers 5.x has `Cohere2ForCausalLM` but lacks the GGUF loading support (missing entries in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, and `GGUF_TO_FAST_CONVERTERS`). The c4ai loader needed a monkey-patch registering the architecture at import time.

**Loader bug 2 (fixed):** Multiple GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 calls this function with `model_to_load=dummy_model` (see modeling_utils.py:4016). The narrow signature silently drops `model_to_load`, causing the chain to fail.

**Compiler bug 1 (Tier A, fixed):** `DynamicSlidingWindowLayer.update()` in transformers produces `full_value_states[:, :, -self.sliding_window+1:, :]` where `sliding_window=4096` but `seq_len=74`. This gives `start=-4095` on a dim-74 tensor. PyTorch silently clamps out-of-bounds negative slice starts to `-dim_size`, but `partition_fx_graph_for_cpu_fallback` in `torch_xla/dynamo_bridge.py` runs the FX graph with XLA tensors to determine device/CPU partitioning. The XLA slice op validates `start ∈ [-dim_size, dim_size-1]` and raises `ValueError: Value out of range` when this precondition is violated, causing the entire compilation to abort.

**Compiler bug 2 (unfixed, this report exits FAIL):** After the Tier A fix, the model compiles and runs on device, but produces PCC=0.9874 vs the CPU bfloat16 baseline (required: 0.99). Measurement: CPU-f32 vs CPU-bf16 PCC = 0.9997, meaning the bfloat16 format itself accounts for only a 0.0003 PCC drop. The hardware TT device shows an additional 0.012 gap, consistent with the consteval-on-host regression (tt-xla issue #1242) but not attributable to bfloat16 format accumulation alone. Per skill rules this gap cannot be papered over by lowering required_pcc; the second bug must be filed as FAIL.

## Fix

**Loader bug 1** — `_patch_transformers_cohere2_gguf()` added to `c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch/loader.py` at import time. Registers `cohere2` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, and `GGUF_TO_FAST_CONVERTERS`. Commit `0dc7044e47` in tt-forge-models on `remediation/c4ai_command_r7b_12_2024_abliterated_gguf-causal_lm-pytorch-single_device-inference`.

**Loader bug 2** — Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated the inner call to `_orig_load_gguf_checkpoint(*args, **kwargs)` across all affected GGUF loaders. Commit `eda47fdf38` in tt-forge-models.

**Compiler bug 1 (Tier A)** — Added a guard in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor`, `args[2]` (start) is a concrete int, and `start < -dim_size`, the start is clamped to `-dim_size` before dispatching to the XLA op. This replicates PyTorch's silent clamping semantics and allows `partition_fx_graph_for_cpu_fallback` to run the graph without raising ValueError. Verified that XLA handles `start=-4095` on a 74-element dim correctly at execution time (PCC=1.0 for isolated slice test). Commit `0dd65deda` in tt-xla on `remediation/c4ai-command-r7b-12-2024-abliterated-gguf-causal-lm-pytorch-single-device-inference`.

**Compiler bug 2 (proposed fix)** — Investigate the consteval-on-host regression (issue #1242) for the Cohere2 model. The fix would ensure that constants evaluated on host use float32, matching the CPU reference behavior. This is a cross-cutting change in the compiler. Alternatively, identify which specific op (LayerNorm, RoPE, attention softmax) accumulates the extra 0.012 PCC drop.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

(Not applicable — second bug unfixed, first Tier A fix committed. Second bug not triaged to Tier B explicitly but likely cross-cutting.)

## Verification
- pytest exit: FAIL (PCC 0.9874 < 0.99)
- Hardware:    n150
- Duration:    555.89s (0:09:15)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `aten.slice.Tensor` out-of-bounds start clamping in `TorchFunctionOverride.__torch_function__`
- `tt-xla/third_party/tt_forge_models/c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch/loader.py` — cohere2 GGUF architecture patch (commit `0dc7044e47`, already on hf-bringup-25)
- Multiple GGUF loader files — narrow `_patched_load_gguf_checkpoint` signatures widened to `(*args, **kwargs)` (commit `eda47fdf38`)
- Reverted forbidden `DynamicSlidingWindowLayer.update` patch (commit `3c674f669f`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0dd65dedafcc1757365b7e943abc14c50d36e4dd |
| tt-forge-models | 3c674f669f3ab508a4dea70201f0ed6851739159 |
