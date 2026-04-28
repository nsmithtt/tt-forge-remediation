# Remediation Summary: alexdenton_qwen3_5_35b_a3b_heretic_gguf/causal_lm/pytorch-35B_A3B_HERETIC_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[alexdenton_qwen3_5_35b_a3b_heretic_gguf/causal_lm/pytorch-35B_A3B_HERETIC_GGUF-single_device-inference]

## Result
FAIL — RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 during flatbuffer graph execution on Blackhole p150b

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
flatbuffer-large-moe-internal-error

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
Fatal Python error: Segmentation fault
```

Failure after applying loader fixes (GGUF architecture fix + batched_mm experts):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  File "venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py", line 826, in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
  File "venv/lib/python3.12/site-packages/torch_xla/torch_xla.py", line 87, in sync
    torch_xla._XLAC._xla_step_marker(
```

## Root cause

**Two distinct bugs were found:**

### Bug 1 (loader — fixed): qwen35moe GGUF architecture not recognized by transformers
Transformers substring-matches 'qwen3moe' within 'qwen35moe', silently converting
model_type to 'qwen3_moe' and breaking Qwen3.5 MoE loading in two ways:
1. Wrong model class selected (Qwen3MoeForCausalLM instead of Qwen3_5MoeForCausalLM)
2. Qwen2MoeTensorProcessor.process() looks up tensor keys without .weight suffix
   but perform_fallback_tensor_mapping adds them with .weight, causing KeyError

Fix patches load_gguf_checkpoint to detect qwen35moe via GGUF header and correct
config model_type/layer_types, and patches get_gguf_hf_weights_map to add both
bare (no-.weight) and separate gate/up entries for MoE expert tensors.

### Bug 2 (loader — fixed): Qwen3_5MoeExperts Python for-loop segfaults XLA
The default Qwen3_5MoeExperts.forward() dispatches experts via a Python for-loop
iterating over a dynamically-sized expert_hit tensor. XLA/torch.compile cannot
statically trace a for-loop whose range depends on a runtime tensor value, causing
a segfault during graph partition probing.

Fix: setting model.config._experts_implementation = "batched_mm" after loading
switches to batched_mm_experts_forward, which uses only static tensor operations
(scatter/gather/matmul) and is fully XLA-compatible.

### Bug 3 (tt-metal — unfixed): INTERNAL error during flatbuffer execution
After the two loader fixes, the full Qwen3.5-35B-A3B model (733 tensors dequantized
to bfloat16, ~70 GB on CPU) compiles successfully (~35-45 minutes) but fails during
FlatbufferLoadedExecutableInstance::execute() with error code kInternal (13). The
error originates in tt::runtime::submit() / ProgramExecutor::execute() and is
identical to the failure pattern seen in byteshape_qwen3_coder_30b_a3b_instruct_gguf
(report dcff2c32).

The root cause of the exception within ProgramExecutor::execute() is not directly
surfaced in the Python traceback. Likely cause: either the compiled flatbuffer binary
exceeds a device execution limit, or device DRAM (p150b ~32 GB) is insufficient to
hold all model weights and intermediate activations for a 35B-parameter model.

## Fix
**Applied in tt-forge-models (loader bugs 1 and 2):**

- `3ae352b217` — Fix alexdenton_qwen3_5_35b_a3b_heretic_gguf: patch GGUF loader for
  qwen35moe architecture (`alexdenton_qwen3_5_35b_a3b_heretic_gguf/causal_lm/pytorch/loader.py`)
- `23e8cb2ac0` — Fix alexdenton_qwen3_5_35b_a3b_heretic_gguf: set batched_mm experts
  to avoid XLA segfault (`alexdenton_qwen3_5_35b_a3b_heretic_gguf/causal_lm/pytorch/loader.py`)

Both commits are on `remediation/alexdenton-qwen3-5-35b-heretic-gguf-segfault` in
tt-forge-models.

**Proposed fix for Bug 3** would live in tt-metal (ProgramExecutor::execute()) or
tt-mlir (flatbuffer serialization/execution). Investigation would need to determine
whether the INTERNAL error is an OOM (model too large for p150b DRAM) or a software
bug in the flatbuffer execution path for large MoE graphs. See byteshape 30B A3B
report (dcff2c32) for a similar failure.

## Tier B justification
<which Tier B indicator applies — internal-error-unknown-mechanism>
The root cause of the exception within ProgramExecutor::execute() is not surfaced in
the Python traceback. The failure is consistent with the byteshape 30B A3B MoE model
(also INTERNAL error) but the root cause is unknown: it could be a DRAM capacity
ceiling or a software bug in the flatbuffer execution path for large MoE graphs.
Diagnosis must precede any fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    7701.08s (2:08:21)
- Tier A attempts: N/A

## Files changed
- `alexdenton_qwen3_5_35b_a3b_heretic_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2dce7069220ce4167b40c4b0160cf900f79b412d |
| tt-forge-models | 23e8cb2ac00d4827b5d71e0f383dfa4be9d19da4 |
