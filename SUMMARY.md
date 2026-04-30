# Remediation Summary: gpt_oss_20b_ilograph_instruct_gguf-causal_lm-pytorch-GPT_OSS_20B_ILOGRAPH_INSTRUCT_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch-GPT_OSS_20B_ILOGRAPH_INSTRUCT_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 during TT device execution (Tier B runtime bug in MoE expert dispatch)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full traceback ends at:
  venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:611: in optimized_mod
      res = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

The original reported failure was `ImportError: Please install torch and gguf>=0.10.0` (missing requirements.txt). In the configured environment gguf was already installed, so the test proceeded further and uncovered a secondary TypeError (`_patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`). After all loader fixes, the terminal failure is INTERNAL: Error code: 13.

## Root cause
GPT-OSS 20B Ilograph Instruct is a Qwen3-MoE-based model. Four loader bugs were found and fixed:
1. Missing `requirements.txt` (gguf>=0.10.0) caused ImportError in CI environments.
2. `_patched_load_gguf_checkpoint` in 27 other loaders lacked `**kwargs`, causing TypeError when those loaders were imported before this one in a pytest session (chain patching bug).
3. `load_shard_spec` referenced dense MLP attributes (`up_proj`, `gate_proj`, `down_proj`) that don't exist on Qwen3-MoE layers; needed MoE-aware guards for `experts.gate_up_proj`, `experts.down_proj`, and `shared_expert.*`.
4. Without `model.config._experts_implementation = "batched_mm"`, the MoE dispatch triggered an XLA segfault.

After all four loader fixes, the model loads and compiles successfully (21 minutes for a 15 GB GGUF / ~44 GB BF16 resident). The TT runtime then raises INTERNAL: Error code: 13 during `_run_cached_graph`, consistent with a device-side failure during MoE expert kernel execution. This is the same Tier B error seen in GLM-4.7/DeepSeek-V2 (also MoE with batched_mm) and is not a loader-level issue.

## Fix
Four loader-layer fixes were applied on branch
`remediation/gpt_oss_20b_ilograph_instruct_gguf-causal_lm-pytorch-GPT_OSS_20B_ILOGRAPH_INSTRUCT_Q4_K_M_GGUF-single_device-inference`
in `tt-xla/third_party/tt_forge_models`:

1. `gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch/requirements.txt` — added (new file) with `gguf>=0.10.0`
2. 27 loader files — `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → added `**kwargs` and forwarded to `_orig_load_gguf_checkpoint`
3. `gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch/loader.py:load_model` — added `model.config._experts_implementation = "batched_mm"` after `from_pretrained`
4. `gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch/loader.py:load_shard_spec` — rewritten to guard on `hasattr(mlp, "experts")` / `hasattr(mlp, "shared_expert")` / `hasattr(layer, "self_attn")`

The terminal INTERNAL: Error code: 13 is in the TT runtime (tt-metal). Proposed fix: investigate MoE expert kernel execution path for unsupported operations or device-to-host transfers triggered by the batched_mm dispatch in Qwen3-MoE. The failure lives in `torch_xla._XLAC._run_cached_graph` — the TT PJRT plugin needs to surface a more specific error code or the tt-metal kernel needs to handle the MoE weight shapes produced by this model.

## Tier B justification
Which indicator applies: `internal-error-unknown-mechanism`

The INTERNAL: Error code: 13 from `_run_cached_graph` lacks a documented TT-specific cause for this model's MoE expert dispatch. Diagnosing the root cause requires instrumenting the tt-metal runtime to surface which kernel or transfer triggers the error — new investigation work, not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    1265.36s (0:21:05)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch/requirements.txt` (added)
- `tt-xla/third_party/tt_forge_models/gpt_oss_20b_ilograph_instruct_gguf/causal_lm/pytorch/loader.py` (load_model + load_shard_spec)
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bfda83b9de88e1ecb69539cc618d951e5f3d96df |
| tt-forge-models | ed6ae594324af3d5e93cf9f333ccfeeb1e798472 |
