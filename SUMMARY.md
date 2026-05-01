# Remediation Summary: lfm2-causal_lm-pytorch-lfm2_24b_a2b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2/causal_lm/pytorch-lfm2_24b_a2b-single_device-inference]

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

After loader fix, secondary failure:
Fatal Python error: Segmentation fault
  at torch_xla/_dynamo/dynamo_bridge.py:762 in partition_fx_graph_for_cpu_fallback
     → tt_torch/torch_overrides.py:34 in __torch_function__
     → (native crash, no Python exception)

## Root cause

**Bug 1 — loader (fixed):** `transformers.integrations.moe.grouped_mm_experts_forward`
branches on `device.type != "cpu"` to choose between float and int input for
`torch.histc`. On XLA device this takes the int path, but XLA ops execute via
`partition_fx_graph_for_cpu_fallback` which runs `torch.histc` on the CPU backend.
PyTorch's CPU `histogram_cpu` kernel does not support integer input, raising:
`NotImplementedError: "histogram_cpu" not implemented for 'Int'`.

Fix: change the condition so that only CUDA uses int; all other devices (CPU, XLA)
use float. Also update `ExpertsInterface._global_mapping["grouped_mm"]` so the
`use_experts_implementation` dispatch picks up the patched function at runtime.

**Bug 2 — compiler stack (unfixed):** After the histc fix, the model compiles multiple
subgraphs successfully but then crashes with a native segfault inside
`partition_fx_graph_for_cpu_fallback` at `dynamo_bridge.py:762`, routed through
`tt_torch/torch_overrides.py:34` → `func(*args, ...)` → native crash. The segfault
occurs consistently (reproduced on two separate runs after device reset). No Python-level
exception is raised before the crash, indicating a null dereference or memory corruption
in the XLA bridge when processing LFM2-MoE's 64-expert graph topology.

## Fix
- `lfm2/causal_lm/pytorch/loader.py`: added `_patch_grouped_mm_experts_forward()` that
  replaces `moe_module.grouped_mm_experts_forward` with a version that uses
  `expert_ids_g.float()` when `device.type != "cuda"`. Called in `load_model()` before
  `from_pretrained`.

**Proposed fix for Bug 2:** The segfault is in the XLA bridge's
`partition_fx_graph_for_cpu_fallback` when tracing LFM2-MoE. The 64-expert MoE topology
generates a complex `argsort → gather → grouped_linear` graph that appears to trigger a
null dereference in `__torch_function__` dispatch through the XLA bridge. Diagnosis
requires native debugging (gdb/lldb) of the `pjrt_plugin_tt.so` to identify the crash
site, then determining whether it is a shape-tracking bug in the FakeTensor path or an
edge case in the XLA bridge's op-to-HLO lowering for LFM2-MoE's unique control flow.
Fix would likely be in `tt-xla`'s dynamo bridge or `torch_overrides.py` (1–3 files,
potentially cross-cutting if it's a shape-dispatch issue).

## Tier B justification
`internal-error-unknown-mechanism`: The crash is a native segfault inside the XLA bridge
with no Python-level exception or error string. The crash site is in native C++ code
called from `partition_fx_graph_for_cpu_fallback` → `torch_overrides.py:34 __torch_function__`.
Root cause diagnosis requires native debugging tools (gdb/lldb on `pjrt_plugin_tt.so`) and
is outside single-PR scope.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    387.04s (first run, histc failure); ~148s (second run, segfault after fix)
- Tier A attempts: N/A

## Files changed
- `lfm2/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e8f55cb7c082b51a87ace9d06dd25460b5d9f0bd |
