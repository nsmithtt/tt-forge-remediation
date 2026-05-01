# Remediation Summary: deepseek_coder_v2_lite_instruct_gguf-causal_lm-pytorch-Coder_V2_Lite_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_coder_v2_lite_instruct_gguf/causal_lm/pytorch-Coder_V2_Lite_Instruct_GGUF-single_device-inference]

## Result
FAIL — INTERNAL Error code: 13 after all loader fixes; pjrt-device-to-host-transfer Tier B bug

## Stack layer
tt-xla

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
The original CI failure on branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-1`:

```
transformers/modeling_gguf_pytorch_utils.py:369: in get_gguf_hf_weights_map
    raise NotImplementedError(
NotImplementedError: Unknown gguf model_type: deepseek_v2 in gguf-py.
```

After all loader fixes, the test fails with:
```
venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:611: in optimized_mod
    res = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two distinct bugs:

**Loader bugs (fixed):**

1. **Missing `get_gguf_hf_weights_map` arch remap**: `transformers.modeling_gguf_pytorch_utils.get_gguf_hf_weights_map` has no `deepseek_v2 → deepseek2` mapping; `gguf-py` uses `"deepseek2"` as the arch name for DeepSeek V2 models. At CI collection time, the `glm_4_7_flash_gguf` loader patches `deepseek2` into `GGUF_SUPPORTED_ARCHITECTURES`, allowing tokenizer loading to succeed, but `get_gguf_hf_weights_map` then raises `NotImplementedError("Unknown gguf model_type: deepseek_v2")`. Fix: patch `get_gguf_hf_weights_map` to substitute `deepseek2` as the arch name when `model_type == "deepseek_v2"`.

2. **MoE expert key mismatch**: `gguf-py` name map returns `ffn_gate_up_exps` (merged, non-existent in GGUF file) instead of the separate `ffn_gate_exps`/`ffn_up_exps` keys. Fix: post-process the weight map to replace the merged key with two real keys and register `Qwen2MoeTensorProcessor` for `deepseek2`.

3. **`q_lora_rank` not set to `None`**: GGUF Q4_K_M stores the absorbed single Q projection; the low-rank MLA form (`q_a_proj`/`q_b_proj`) doesn't exist in GGUF. Fix: `config.q_lora_rank = None`.

4. **`grouped_mm` MoE histc failure**: `grouped_mm_experts_forward` uses `torch.histc` whose XLA lowering fails. Fix: `config._experts_implementation = "batched_mm"`.

5. **Complex-valued RoPE**: `torch.polar` creates complex `freqs_cis` tensors that TT cannot handle at XLA subgraph boundaries. Fix: replace `DeepseekV2RotaryEmbedding.forward` and `apply_rotary_emb` with real-arithmetic equivalents returning stacked `[batch, seq, d/2, 2]` tensors.

6. **Missing `requirements.txt`**: `gguf>=0.10.0` not listed.

7. **`_patched_load_gguf_checkpoint` missing `**kwargs`**: In a full pytest session other loaders patch `load_gguf_checkpoint` without forwarding `model_to_load=` kwarg from transformers 5.x, causing `TypeError`. Fix: forward `**kwargs` through patched chain.

**Compiler-stack bug (unfixed, Tier B):**
After all loader fixes, the compiled TT graph fails at runtime with `INTERNAL: Error code: 13`. This is the known `pjrt-device-to-host-transfer` bug: a tensor operation inside the DeepSeek V2 MoE forward pass (likely the top-k routing producing integer index tensors) triggers a device-to-host transfer that the PJRT transport layer cannot fulfill.

## Fix
**Loader fixes** applied in `tt_forge_models` on branch `remediation/deepseek_coder_v2_lite_instruct_gguf-causal_lm-pytorch-Coder_V2_Lite_Instruct_GGUF-single_device-inference` (commit `92cdc50219`):

- `deepseek_coder_v2_lite_instruct_gguf/causal_lm/pytorch/loader.py` — all seven fixes listed above.
- `deepseek_coder_v2_lite_instruct_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`.

**Proposed fix for the Tier B bug:**
The PJRT buffer-instance device-to-host transfer path in `tt-xla` does not support returning integer tensors from compiled TT graphs (or transferring scalar/index tensors mid-graph for control flow). Fixing this requires implementing the missing device→host tensor-copy path in the PJRT implementation, which is new infrastructure touching `pjrt_implementation/src/api/`.

## Tier B justification
`new-infrastructure`: Fixing Error code: 13 requires implementing PJRT device→host tensor transfer support, which is a missing infrastructure capability in the TT PJRT plugin, not a scoped pattern fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    786.53s (0:13:06)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/deepseek_coder_v2_lite_instruct_gguf/causal_lm/pytorch/loader.py` (loader fixes)
- `tt_forge_models/deepseek_coder_v2_lite_instruct_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b39543ebef719ab9355d08d165d922d451882ee9 |
| tt-forge-models | 92cdc502199b726a27e59b4d3ee35f533e302a84 |
