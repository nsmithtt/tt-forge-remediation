# Remediation Summary: byteshape_qwen3_coder_30b_a3b_instruct_gguf/causal_lm/pytorch-30B_A3B_Instruct_GGUF-single_device-inference

## Skill version
9

## Test
tests/runner/test_models.py::test_all_models_torch[byteshape_qwen3_coder_30b_a3b_instruct_gguf/causal_lm/pytorch-30B_A3B_Instruct_GGUF-single_device-inference]

## Result
FAIL — RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 during flatbuffer graph execution on Blackhole p150b

## Failure
Original CI failure:
```
Fatal Python error: Segmentation fault
```

Failure after applying loader fix (batched_mm experts):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  File "venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py", line 611, in optimized_mod
    res = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
```

## Root cause

**Two distinct bugs were found:**

### Bug 1 (loader layer — fixed): Segfault from Python for-loop in Qwen3MoeExperts
The default `Qwen3MoeExperts.forward()` dispatches experts via a Python for-loop iterating over a dynamically-sized `expert_hit` tensor. XLA/torch.compile cannot statically trace a for-loop whose range depends on a runtime tensor value, causing a segfault during graph partition probing.

**Fix:** Setting `model.config._experts_implementation = "batched_mm"` after loading switches to `batched_mm_experts_forward`, which uses only static tensor operations (scatter/gather/matmul) and is fully XLA-compatible.

### Bug 2 (runtime layer — unfixed): INTERNAL error during flatbuffer execution
After applying the batched_mm fix, the full 64-layer MoE model (30B parameters, 579 tensors dequantized to bfloat16, ~60GB on CPU) compiles successfully (~22 minutes) but fails during `FlatbufferLoadedExecutableInstance::execute()` with error code `kInternal` (13). The error originates in `tt::runtime::submit()` / `ProgramExecutor::execute()`.

The error path is:
```
tt::pjrt::FlatbufferLoadedExecutableInstance::execute()
  → invoke_noexcept(tt::runtime::submit, ...)
    → ProgramExecutor::execute() throws exception
  → invoke_noexcept returns nullopt → kInternal
```

The root cause of the exception within `ProgramExecutor::execute()` is not directly surfaced in the Python traceback. The compiled graph includes all 64 layers × MoE experts passed as inputs, creating a very large flatbuffer binary and a large number of device tensors. The failure is consistent across two independent test runs with a device reset between them.

Likely cause: Either the compiled flatbuffer binary exceeds a device execution limit, or device DRAM is insufficient to hold all model weights and intermediate activations simultaneously at the BFP16 precision level needed for execution.

## Fix

**Applied in tt-forge-models (`a74406c157`):**

Added the `byteshape_qwen3_coder_30b_a3b_instruct_gguf` loader to the main branch with:
- `loader.py`: Full model loader with `model.config._experts_implementation = "batched_mm"` set after loading
- `requirements.txt`: `gguf>=0.10.0`
- `__init__.py`

This is not a forbidden workaround: it selects between two equivalent forward implementations in Qwen3MoE, choosing the one that uses static tensor operations instead of a Python for-loop. No layers are trimmed, no parameters are moved to CPU, no inputs are changed.

**Remaining runtime bug (unfixed):**

The `FlatbufferLoadedExecutableInstance::execute()` failure for large MoE models needs investigation in tt-mlir/tt-metal. Proposed investigation:
1. Enable `TT_RUNTIME_DEBUG=1` to capture device memory state before/after submit
2. Check if the compiled flatbuffer binary size exceeds device program size limits
3. Verify available DRAM vs. total model weight tensor size at BFP16 precision
4. Investigate whether tt-mlir needs to emit weight streaming operations for models > 48GB

## Verification
pytest exit status: FAILED
Wall-clock duration: 22 minutes 59 seconds (loading + compilation + execution failure)
Hardware: Blackhole p150b (8 GDDR6 channels enabled)

Both runs produced identical `INTERNAL: Error code: 13` failures at the same call site.

## Files changed
- `byteshape_qwen3_coder_30b_a3b_instruct_gguf/causal_lm/pytorch/__init__.py` (new)
- `byteshape_qwen3_coder_30b_a3b_instruct_gguf/causal_lm/pytorch/loader.py` (new)
- `byteshape_qwen3_coder_30b_a3b_instruct_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a7e87bd7b8083a69be7f2206d331ddb7e81f9574 |
| tt-forge-models | a74406c157aeb73bd25f3241181077dd11e7e824 |
