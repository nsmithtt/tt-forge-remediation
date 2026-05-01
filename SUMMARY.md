# Remediation Summary: liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf-causal_lm-pytorch-21B_A3B_Thinking_Gemini_2.5_Pro_Distill_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf/causal_lm/pytorch-21B_A3B_Thinking_Gemini_2.5_Pro_Distill_GGUF-single_device-inference]

## Result
FAIL — Segfault in TT backend C++ during partition_fx_graph_for_cpu_fallback after Tier A histc fix

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
tt-backend-sigsegv-fx-partition-ernie4_5moe

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
NotImplementedError: "histogram_cpu" not implemented for 'Int'

While executing %histc : [num_users=1] = call_function[target=torch.ops.aten.histc.default](args = (%_to_copy_40, 64, 0, 63), kwargs = {})
Original traceback:
  File ".../transformers/integrations/moe.py", line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
  File ".../tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))
```

After Tier A histc fix, new failure:
```
Fatal Python error: Segmentation fault

Current thread (most recent call first):
  File ".../torch/_ops.py", line 841 in __call__
  File ".../tt_torch/torch_overrides.py", line 39 in __torch_function__
  File ".../torch/_ops.py", line 841 in __call__
  File ".../torch/fx/interpreter.py", line 336 in call_function
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File ".../torch_xla/_dynamo/dynamo_bridge.py", line 859 in extract_compiled_graph_helper
```

## Root cause
Two bugs stacked:

**Bug 1 (loader / tt-xla, fixed):** `transformers/integrations/moe.py` calls `torch.histc(histc_input, ...)` where `histc_input` is an integer tensor (`expert_ids_g.int()`) when the device type is not CPU. The original `torch_overrides.py` fix checked `func.__name__ == "histc"` but when the FX interpreter replays the graph it calls `torch.ops.aten.histc.default`, whose `__name__` is `"histc.default"` — so the check never matched and the integer tensor reached the CPU histogram kernel, which does not support integer dtypes.

**Bug 2 (tt-xla/tt-metal, Tier B):** After the histc fix allows the model to proceed past token-counting, the TT backend C++ crashes with a SIGSEGV inside `partition_fx_graph_for_cpu_fallback` → `extract_compiled_graph_helper`. The specific C++ op or allocation that faults is unknown without a native debugger. This is the same class of bug as `tt-backend-sigsegv-fx-partition-lfm2moe`.

## Fix
**Loader fixes (tt_forge_models, 4ce235abe1):**
- `liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf/causal_lm/pytorch/loader.py` — registers `ernie4_5-moe` in GGUF tables (GGUF_SUPPORTED_ARCHITECTURES, GGUF_TO_TRANSFORMERS_MAPPING, TENSOR_PROCESSORS, GGUF_TO_FAST_CONVERTERS), remaps `ernie4_5-moe` → `ernie4_5_moe` model_type, patches get_gguf_hf_weights_map for reverse remap
- `liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf/causal_lm/pytorch/requirements.txt` — adds `gguf>=0.10.0`
- `**kwargs` forwarding fix in `_patched_load_gguf_checkpoint` chain

**Tier A compiler-stack fix (tt-xla, 4cf25f39a):**
- `python_package/tt_torch/torch_overrides.py` — changed `func.__name__ == "histc"` to `"histc" in getattr(func, "__name__", "")` so the int→float cast fires both for `torch.histc` (eager, `__name__=="histc"`) and `torch.ops.aten.histc.default` (FX interpreter, `__name__=="histc.default"`).

**Proposed fix for Tier B bug:**
The segfault in `partition_fx_graph_for_cpu_fallback` requires C++ debugging to identify the crashing op. Likely lives in `torch_xla/_dynamo/dynamo_bridge.py` or the TT PJRT plugin's FX graph partitioning code. The fix would need to identify which node in the ERNIE 4.5 MoE graph triggers a NULL dereference or invalid memory access during CPU-fallback partitioning.

## Tier B justification
`internal-error-unknown-mechanism` — The SIGSEGV occurs deep in C++ within `partition_fx_graph_for_cpu_fallback`; the crashing op and root cause are unknown without a native debugger. No Python-level stack frame identifies the faulty node.

## Verification
- pytest exit: FAIL (segfault, exit 139)
- Hardware:    blackhole-p150b
- Duration:    ~38 minutes total (two test runs of ~17 min each)
- Tier A attempts: 1

## Files changed
**tt_forge_models (remediation branch 4ce235abe1):**
- `liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf/causal_lm/pytorch/loader.py`
- `liontix_ernie_4_5_21b_a3b_thinking_gemini_2_5_pro_distill_gguf/causal_lm/pytorch/requirements.txt`

**tt-xla (remediation branch 4cf25f39a):**
- `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4cf25f39af4a2b495c23cc113e5d002ca4cd82eb |
| tt-forge-models | 4ce235abe1e54ff43183c336972d7dd4b25cc5ef |
