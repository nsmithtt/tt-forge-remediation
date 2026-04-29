# Remediation Summary: bartowski_liquidai_lfm2_24b_a2b_gguf-causal_lm-pytorch-LiquidAI_LFM2_24B_A2B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_liquidai_lfm2_24b_a2b_gguf/causal_lm/pytorch-LiquidAI_LFM2_24B_A2B_GGUF-single_device-inference]

## Result
FAIL — loader fix applied (histogram int→float for XLA); test then crashes with segfault in partition_fx_graph_for_cpu_fallback during model execution

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
moe-histc-int-xla-fallback, pjrt-dynamo-partition-cpu-fallback-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   NotImplementedError: "histogram_cpu" not implemented for 'Int'

While executing %histc : [num_users=1] = call_function[target=torch.ops.aten.histc.default](args = (%_to_copy_12, 64, 0, 63), kwargs = {})
Original traceback:
  File ".../transformers/models/lfm2_moe/modeling_lfm2_moe.py", line 665, in forward
    hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
  File ".../transformers/models/lfm2_moe/modeling_lfm2_moe.py", line 227, in forward
    final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
  File ".../transformers/integrations/moe.py", line 373, in forward
    return experts_forward(self, *args, **kwargs)
  File ".../transformers/integrations/moe.py", line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)

## Root cause

**Bug 1 — loader (fixed):** `transformers.integrations.moe.grouped_mm_experts_forward`
branches on `device.type != "cpu"` to decide whether to call `expert_ids_g.int()` for
`torch.histc`. On XLA device this takes the int path, but XLA ops execute via
`partition_fx_graph_for_cpu_fallback` which runs `torch.histc` on CPU. PyTorch's CPU
`histogram_cpu` kernel does not support integer input, raising:
`NotImplementedError: "histogram_cpu" not implemented for 'Int'`.

The LFM2-24B-A2B GGUF also uses the lfm2moe architecture string which is not registered
in transformers' GGUF tables (GGUF_SUPPORTED_ARCHITECTURES, TENSOR_PROCESSORS, GGUF_TO_TRANSFORMERS_MAPPING),
and `load_gguf_checkpoint` was not forwarding `**kwargs` (missing `model_to_load`
kwarg introduced in transformers 5.x).

**Bug 2 — compiler stack (unfixed):** After the loader fixes, the model compiles multiple
subgraphs successfully but then segfaults inside `partition_fx_graph_for_cpu_fallback`
during `run_node` → `__torch_function__` (tt_torch/torch_overrides.py:34). The crash
is a native `Segmentation fault` with no Python-level exception, indicating a null
pointer or memory corruption in the PJRT plugin or XLA dynamo bridge when handling
the lfm2_moe graph.

## Fix

**Fix 1 — loader (applied, tt_forge_models remediation branch):**

1. Registered `lfm2moe` in `GGUF_SUPPORTED_ARCHITECTURES`, `TENSOR_PROCESSORS`, and
   `GGUF_TO_TRANSFORMERS_MAPPING["config"]` at import time, with the full config key
   mapping required by the LFM2-MoE architecture, and in `GGUF_TO_FAST_CONVERTERS`.

2. Wrapped `load_gguf_checkpoint` with `_apply_lfm2moe_load_patches()` called in
   `_load_tokenizer`, `load_model`, and `load_config`. The wrapper:
   - Uses `_find_base_load_gguf()` (sys.modules search) to find the original function,
     bypassing broken wrappers from other loaders that lack `**kwargs`.
   - Remaps `model_type: lfm2moe → lfm2_moe` in the returned config dict.
   - Translates `num_key_value_heads` list to scalar + derives `layer_types`.

3. Patched `grouped_mm_experts_forward` to use `expert_ids_g.float()` when
   `device.type != "cuda"` (XLA and CPU both need float for `torch.histc`).

4. Added `requirements.txt` with `gguf>=0.10.0`.

**Fix 2 — segfault (proposed, not attempted):**
Investigate `tt_torch/torch_overrides.py` and the `partition_fx_graph_for_cpu_fallback`
path for null-pointer or use-after-free when processing lfm2_moe MoE expert ops.
Likely requires debugging the C++ pjrt_plugin_tt.so with the lfm2_moe graph shape.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

The second failure is a `Segmentation fault` in native C++ code (PJRT plugin /
torch-xla dynamo bridge) with no Python exception and no clear mechanism. Diagnosing
requires C++ debugging of the pjrt_plugin_tt.so with the lfm2_moe computation graph,
touching potentially multiple files across tt-xla and tt-metal. The crash site
(`torch_overrides.py:34` → `torch._ops.__call__`) provides no further signal about
which op or which null pointer triggered the fault.

## Verification
- pytest exit: FAIL (segmentation fault after loader fix)
- Hardware:    n150
- Duration:    1364.74s (0:22:44) for baseline run; second run crashed mid-execution
- Tier A attempts: N/A (loader fix; Tier B for compiler crash)

## Files changed
- `bartowski_liquidai_lfm2_24b_a2b_gguf/causal_lm/pytorch/loader.py` — full rewrite with lfm2moe GGUF support, load_gguf_checkpoint patching, grouped_mm histc fix
- `bartowski_liquidai_lfm2_24b_a2b_gguf/causal_lm/pytorch/requirements.txt` — new file, adds `gguf>=0.10.0`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 802bdad7eae0aa9f0f3a49c3f4756cb33f03061f |
| tt-forge-models | 09d8a81cdd551f196ae98641072c7fa2a167df76 |
