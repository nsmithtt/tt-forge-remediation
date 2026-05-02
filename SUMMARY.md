# Remediation Summary: mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf-causal_lm-pytorch-24B_A3B_CLAUDE_OPUS_GEMINI_3_1_PRO_REASONING_DISTILLED_HERETIC_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf/causal_lm/pytorch-24B_A3B_CLAUDE_OPUS_GEMINI_3_1_PRO_REASONING_DISTILLED_HERETIC_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — SIGSEGV in partition_fx_graph_for_cpu_fallback (tt-xla dynamo bridge, Tier B)

## Stack layer
loader, tt-xla

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
Original reported failure:
```
ImportError: Please install torch and gguf>=0.10.0
```

After loader fixes, terminal failure:
```
Fatal Python error: Segmentation fault

Current thread 0x00007f03bf9ff640 (most recent call first):
  File ".../torch/_ops.py", line 841 in __call__
  File ".../tt_torch/torch_overrides.py", line 34 in __torch_function__
  File ".../torch/_ops.py", line 841 in __call__
  File ".../torch/fx/interpreter.py", line 336 in call_function
  File ".../torch/fx/interpreter.py", line 256 in run_node
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File ".../torch/fx/interpreter.py", line 174 in run
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 737 in extract_compiled_graph
  File ".../tt_torch/backend/backend.py", line 215 in _call_experimental_compile
  ...
  File ".../modeling_qwen3_5_moe.py", line 1487 in torch_dynamo_resume_in_forward_at_1487
```

## Root cause

**Loader layer (4 bugs fixed):**

1. **Missing gguf dependency** (`requirements.txt`): The requirements file lacked `gguf>=0.10.0`, causing the original ImportError.

2. **Session contamination — narrow-sig wrappers** (`load_gguf_checkpoint`): pytest imports model loaders alphabetically at collection time. Other loaders (gpt_oss, onion008-style models) replace `gguf_utils.load_gguf_checkpoint` with narrow-signature wrappers that do not accept the `model_to_load` keyword argument added in transformers 5.2.0. The fix traverses wrapper chains via globals (`_orig_load_gguf_checkpoint`) and closure variables (`orig_load`, `real_load`) to find the real transformers function, then re-wraps it.

3. **ffn_gate_up_exps key mismatch**: gguf-py 0.18 maps `gate_up_proj` to the merged GGUF key `ffn_gate_up_exps`, but the actual GGUF file stores separate `ffn_gate_exps` and `ffn_up_exps` tensors. `Qwen2MoeTensorProcessor.process()` expects these separate keys when constructing `gate_up_proj`. The fix post-processes the weight map to replace the merged key with both separate keys.

4. **Conv1d weight shape mismatch**: The Qwen3.5 MoE hybrid architecture has GatedDeltaNet layers with depthwise `ssm_conv1d` layers. GGUF stores the weight as 2D `(C, K)` but `nn.Conv1d(groups=C)` requires 3D `(C, 1, K)`. The fix unsqueezes dimension 1 on any `*.conv1d.weight` tensor with `ndim==2` after GGUF loading.

**Compiler layer (Tier B):**

After all loader fixes, the model loads successfully and enters `torch.compile` with the TT backend. The dynamo bridge in tt-xla (`partition_fx_graph_for_cpu_fallback`) crashes with a fatal SIGSEGV while probing operations (including Conv1d) for device compatibility via `TorchFunctionOverride`. This is the same crash pattern observed for `enzgamers_qwen3_5_35b_a3b_gguf` — Conv1d operations in the GatedDeltaNet (SSM) layers trigger XLA probing in `partition_fx_graph_for_cpu_fallback` that causes a segfault. The mechanism is unknown and requires deep diagnosis of the XLA dynamo bridge and tt_torch device probing path.

## Fix

**Loader fixes in `tt-forge-models`, branch `remediation/mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf-...`:**

- `mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`
- `mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf/causal_lm/pytorch/loader.py` — added complete `_patch_transformers_qwen35moe_gguf()` with:
  - qwen35moe architecture registration in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `TENSOR_PROCESSORS`, and `GGUF_TO_FAST_CONVERTERS`
  - Wrapper chain traversal to find real `load_gguf_checkpoint` via globals and closure inspection
  - `patched_load_gguf_checkpoint` that rewrites `model_type=qwen35moe` → `qwen3_5_moe_text` and builds `layer_types` list from `full_attention_interval`
  - Conv1d weight unsqueeze for depthwise weights stored 2D in GGUF
  - `patched_get_gguf_hf_weights_map` to replace merged `ffn_gate_up_exps` key with separate `ffn_gate_exps` + `ffn_up_exps` keys

**No fix for Tier B compiler-stack bug.**

## Tier B justification
`internal-error-unknown-mechanism`

The SIGSEGV occurs inside `partition_fx_graph_for_cpu_fallback` in the tt-xla dynamo bridge while probing Conv1d operations from the Qwen3.5 GatedDeltaNet layers. The crash has no error string — it is a fatal signal with unknown mechanism. Fixing it would require understanding and modifying the XLA device-probing path in the dynamo bridge, which is cross-cutting infrastructure (not a single-function fix).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1214.45s (model loading ~20 min)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf/causal_lm/pytorch/requirements.txt` (created)
- `tt-forge-models`: `mradermacher_qwen3_5_24b_a3b_claude_opus_gemini_3_1_pro_reasoning_distilled_heretic_gguf/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6acb8b1c105e076f9e7448d7402616613c55a839 |
| tt-forge-models | 3a3369298fdcfe4f7d85678ff9102ea3d39f8317 |
