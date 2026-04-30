# Remediation Summary: deepseek_v2_lite_gguf-causal_lm-pytorch-DeepSeek_V2_Lite_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_v2_lite_gguf/causal_lm/pytorch-DeepSeek_V2_Lite_GGUF-single_device-inference]

## Result
FAIL — INTERNAL: Error code: 13 at torch_xla._XLAC._run_cached_graph after seven loader fixes applied; Tier B compiler/runtime bug

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
tt-runtime-internal-error-code-13

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original stated failure: raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
Actual reproduced failure on hf-bringup-7: KeyError: 'deepseek_v2' in GGUF_TO_FAST_CONVERTERS (iquest loader patches convert_gguf_tokenizer; deepseek_v2 not registered)
After all loader fixes: RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Seven loader bugs prevented the model from loading and running on TT silicon:

1. **Missing GGUF tokenizer converter**: `GGUF_TO_FAST_CONVERTERS` has no `deepseek_v2` entry in transformers 5.x. The `iquest_coder_v1_40b_base_gguf` loader patches `convert_gguf_tokenizer` at import; when it calls through to the original for `deepseek_v2`, it hits `KeyError`.

2. **Missing requirements.txt**: `gguf>=0.10.0` was not listed as a dependency.

3. **kwargs forwarding in 26 patched loaders**: `_patched_load_gguf_checkpoint` wrappers across 26 loaders did not accept `**kwargs`, causing `TypeError` in full pytest sessions when transformers 5.x passes `model_to_load=` keyword argument.

4. **No deepseek_v2→deepseek2 arch mapping**: `get_gguf_hf_weights_map` in transformers 5.x does not map `model_type="deepseek_v2"` to the gguf-py arch name `"deepseek2"`, causing `NotImplementedError`.

5. **MoE expert weight mismatch**: gguf-py's name map returns the non-existent `ffn_gate_up_exps` merged key; the actual GGUF file stores separate `ffn_gate_exps` and `ffn_up_exps`. `Qwen2MoeTensorProcessor` was not registered for `deepseek2`.

6. **q_lora_rank mismatch**: The GGUF absorbs the MLA low-rank Q projection into a single `attn_q` weight; leaving `q_lora_rank` non-None causes the model to expect split `q_a_proj`/`q_b_proj` that don't exist.

7. **grouped_mm MoE + real-arithmetic RoPE**: `grouped_mm_experts_forward` uses `torch.histc` whose XLA lowering fails. `torch.polar` produces complex types that either cause CB overflow or PJRT buffer errors at the XLA subgraph boundary.

After all seven loader fixes the model loads, tokenizes, and compiles. During execution on TT silicon, `_run_cached_graph` fails with `INTERNAL: Error code: 13`. This is an opaque runtime error from the TT PJRT backend with no further diagnostic output — the specific kernel or op causing the failure cannot be determined from Python-level tracing alone.

## Fix
All seven loader fixes were applied in `deepseek_v2_lite_gguf/causal_lm/pytorch/loader.py` and related files (26 other loaders for the **kwargs fix):
- `third_party/tt_forge_models`: `remediation/deepseek_v2_lite_gguf-causal_lm-pytorch-DeepSeek_V2_Lite_GGUF-single_device-inference` branch, commit `e9c8ad9c`
- `tt-xla`: `remediation/deepseek_v2_lite_gguf-causal_lm-pytorch-DeepSeek_V2_Lite_GGUF-single_device-inference` branch, commit `267939299`

For the Tier B `INTERNAL: Error code: 13` bug, the proposed investigation path is:
- Enable `TT_LOGGER_LEVEL=DEBUG` or `TTNN_LOGGER_LEVEL=DEBUG` to get device-level error detail
- Identify which op/kernel fails at execution time
- The previous remediation run (Apr 30 00:02) observed `CB overflow (5878784 B > 1572864 B L1)` at the RoPE subgraph; the current run's error code 13 may be the same root cause now manifesting differently after the real-arithmetic RoPE fix altered subgraph shape

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

The error `INTERNAL: Error code: 13` at `_run_cached_graph` provides no information about which kernel or op failed. The fix mechanism is unknown — diagnosis (running with debug logging to identify the failing kernel and CB/memory state) must precede any fix attempt.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    855.78s (0:14:15)
- Tier A attempts: N/A

## Files changed
- `deepseek_v2_lite_gguf/causal_lm/pytorch/loader.py` — seven loader fixes
- `deepseek_v2_lite_gguf/causal_lm/pytorch/requirements.txt` — new file with gguf>=0.10.0
- 26 other GGUF loader files — `**kwargs` fix for `_patched_load_gguf_checkpoint`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 267939299424354039012fecb4108ca92f6be9ac |
| tt-forge-models | e9c8ad9c106d592bab2ce344a57a626351936d92 |
