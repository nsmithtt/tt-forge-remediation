# Remediation Summary: mradermacher_iflow_rome_gguf-causal_lm-pytorch-iFlow_ROME_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_iflow_rome_gguf/causal_lm/pytorch-iFlow_ROME_GGUF-single_device-inference]

## Result
FAIL — Fatal Python error: Segmentation fault in tt-xla extract_compiled_graph during partition_fx_graph_for_cpu_fallback; Tier B, root cause unknown without C-level debugging

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
tt-xla-partition-segfault-unknown-op

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

Thread 0x0000704f7c7e2140 (most recent call first):
  File "torch/_ops.py", line 841 in __call__
  File "tt_torch/torch_overrides.py", line 34 in __torch_function__
  File "torch/_ops.py", line 841 in __call__
  File "torch/fx/interpreter.py", line 336 in call_function
  File "torch/fx/interpreter.py", line 256 in run_node
  File "torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File "torch/fx/interpreter.py", line 174 in run
  File "torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File "torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
  File "torch_xla/_dynamo/dynamo_bridge.py", line 737 in extract_compiled_graph
  File "tt_torch/backend/backend.py", line 215 in _call_experimental_compile

## Root cause
Two bugs were found.

**Loader bug (fixed):** Multiple GGUF model loaders in tt-forge-models monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a
function whose signature is `(gguf_path, return_tensors=False)`. In transformers 5.2.0 this
function was extended to `(gguf_checkpoint_path, return_tensors=False, model_to_load=None)`,
and the caller in `modeling_utils.py:4016` always passes `model_to_load=dummy_model`. When
pytest collects all tests it imports every loader module, leaving the stale patch active. Any
subsequent GGUF model test (including this one) then fails with `TypeError:
_patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fixed
by adding `model_to_load=None` to the 26 affected patched-function signatures and forwarding
the argument to `_orig_load_gguf_checkpoint`.

**Compiler-stack bug (Tier B, unfixed):** After the loader fix the model loads successfully
(579 GGUF tensors de-quantized to bfloat16, ~20 minutes). Compilation then crashes with a
segmentation fault inside `bridge.extract_compiled_graph` → `partition_fx_graph_for_cpu_fallback`
→ `run_node`. The crash originates in `torch._ops.__call__` called from
`TorchFunctionOverride.__torch_function__` (tt_torch/torch_overrides.py:34) during op probing.
`partition_fx_graph_for_cpu_fallback` runs each FX graph node to determine whether it can
execute on the TT device; some op in the iFlow-ROME graph triggers a C-level null pointer or
invalid memory access in the PJRT/XLA backend. Python-level tracing ends at `func(*args,
**kwargs)` — no higher-level Python exception is raised before the process dies.
The specific failing op is unknown without C-level (gdb/lldb) instrumentation.

## Fix
**Loader fix committed:** Branch
`remediation/mradermacher_iflow_rome_gguf-causal_lm-pytorch-iFlow_ROME_GGUF-single_device-inference`
in `tt-forge-models` (commit `ce0497bdd1d6a96e732b1b10b3823f9754f71f5b`).

Changed 26 files:
- `*/causal_lm/pytorch/loader.py` in the following model directories:
  qwen_3_5_imatrix_gguf, mradermacher_vilm_0_8b_sft_gguf, mradermacher_qwen3_5_27b_gguf,
  mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_4b_abliterated_i1_gguf,
  mradermacher_qwen3_5_4b_unfiltered_gguf, mradermacher_qwen3_5_27b_tainted_heresy_gguf,
  mradermacher_qwen3_5_4b_ara_heresy_v2_gguf, mradermacher_qwen3_5_4b_gabliterated_gguf,
  mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf, mradermacher_qwen3_5_9b_abliterated_i1_gguf,
  mradermacher_qwen3_5_4b_unredacted_max_gguf, mradermacher_luna_qwen3_5_27b_v5_i1_gguf,
  mradermacher_bartleby_qwen3_5_4b_gguf, mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf,
  mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf,
  mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf,
  tvall43_qwen3_5_2b_heretic_v3b_i1_gguf, tvall43_qwen3_5_4b_heretic_v2_i1_gguf,
  unified_reward_flex_qwen35_27b_gguf, gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf,
  gpt_oss_swallow_20b_rl_v0_1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf,
  dmind_3_mini_i1_gguf, daniloreddy_qwen3_5_0_8b_gguf, bartowski_coniccat_qwen3_5_27b_writer_gguf

Each file: `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` →
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None)` and
`_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` →
`_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)`

**Proposed compiler fix:** Instrument `partition_fx_graph_for_cpu_fallback` to catch the
crashing op via C-level debugging (gdb with a SIGSEGV handler), identify the specific op and
tensor types causing the crash, then either guard that op in `tt_torch/torch_overrides.py` or
fix the corresponding lowering in tt-metal/tt-mlir.

## Tier B justification
internal-error-unknown-mechanism — the segfault terminates the Python process before any
Python exception is raised; the failing op in the iFlow-ROME FX graph is not identified
without C-level (gdb) instrumentation.

## Verification
- pytest exit: FAIL (Fatal Python error: Segmentation fault)
- Hardware:    n150
- Duration:    ~96 minutes to load + crash (model loading: ~20 min, compilation crash: ~76 min elapsed)
- Tier A attempts: N/A

## Files changed
tt-forge-models (26 loader files — see Fix section above)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ce0497bdd1d6a96e732b1b10b3823f9754f71f5b |
