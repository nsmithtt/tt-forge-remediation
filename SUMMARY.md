# Remediation Summary: davidau_openai_gpt_oss_20b_neo_gguf/causal_lm/pytorch-NEO_IQ4_NL-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[davidau_openai_gpt_oss_20b_neo_gguf/causal_lm/pytorch-NEO_IQ4_NL-single_device-inference]

## Result
FAIL — segfault in torch_xla dynamo bridge during FX graph partitioning of the 20B model after loader TypeError was fixed

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
tt-xla-dynamo-bridge-segfault-large-model

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

Current thread 0x00007f7d33dbf740 (most recent call first):
  File "torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File "torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File "torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
  File "torch_xla/_dynamo/dynamo_bridge.py", line 737 in extract_compiled_graph
  File "tt_torch/backend/backend.py", line 215 in _call_experimental_compile
  File "tt_torch/backend/backend.py", line 225 in __call__

## Root cause
Two failures were uncovered in sequence.

**First failure (fixed — loader layer):** Test discovery imports all GGUF model loaders, 26 of which monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` function. A recent transformers update changed the call site in `modeling_utils.py:4016` to pass `model_to_load=dummy_model` as a keyword argument. Because the patched functions did not accept `model_to_load`, any GGUF model loaded after one of those 26 loaders was imported raised `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fixed by adding `model_to_load=None` to the 26 patched function signatures and forwarding the argument to the original call.

**Second failure (unfixed — tt-xla):** After the loader fix, the model loads successfully (~40 GB in bfloat16 for a 20 B parameter model) and proceeds to torch.compile with the "tt" backend. The dynamo bridge's `partition_fx_graph_for_cpu_fallback` runs the full FX graph once through `UnsupportedNodesCollector.run_node` to detect unsupported ops. During this execution — which materialises XLA tensors representing 20 B parameters — the process crashes with a segfault inside a C extension module. The exact mechanism (null pointer dereference from device OOM, buffer overflow in the graph interpreter, or race condition in the PJRT buffer management) cannot be determined without a native debugger. The crash reproduces consistently.

## Fix
**Loader fix (committed):** Updated 26 loaders in `tt_forge_models` on branch `remediation/davidau-openai-gpt-oss-20b-neo-gguf-causal-lm-pytorch-neo-iq4-nl-single-device-inference`:
- Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):`
- Changed `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)`

**Segfault fix (not attempted — Tier B):** The proposed fix would be to add graceful OOM detection in `torch_xla/_dynamo/dynamo_bridge.py`'s `UnsupportedNodesCollector.run_node` (or in the tt backend's `extract_compiled_graph_helper`) so that when the device or host memory is exhausted while processing a very large model, a clean Python exception is raised rather than crashing the interpreter. This would live in the tt-xla PJRT plugin and/or the torch_xla dynamo bridge.

## Tier B justification
internal-error-unknown-mechanism — The segfault occurs in a C extension module (`torch._C`, `pjrt_plugin_tt.so`, or torch_xla's XLAC) during `run_node` in the FX graph interpreter. Without a native debugger or ASAN output, the precise crash site and mechanism are unknown. The fix hypothesis (add OOM guards in the dynamo bridge) is plausible but unverified, and it may require changes to torch_xla internals outside this repo's scope.

## Verification
- pytest exit: FAIL (Fatal Python error: Segmentation fault)
- Hardware:    blackhole-p150b
- Duration:    ~16:37 wall-clock (0:16:37)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 536b650580315259d2efd1c345d78c4853b50066 |
| tt-forge-models | ac37ec7def5964711f9bc24bd658fa77cb136e51 |
