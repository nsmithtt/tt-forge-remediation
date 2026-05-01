# Remediation Summary: huihui_glm_4_7_flash_abliterated-causal_lm-pytorch-4.7_Flash_abliterated-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_glm_4_7_flash_abliterated/causal_lm/pytorch-4.7_Flash_abliterated-single_device-inference]

## Result
FAIL — segfault in partition_fx_graph_for_cpu_fallback after histc Tier A fix; Tier B compiler bug blocks further progress

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
grouped-mm-cpu-fallback-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `NotImplementedError: "histogram_cpu" not implemented for 'Int'` while executing `%histc` in `grouped_mm_experts_forward` (transformers/integrations/moe.py:271). After Tier A histc fix: `Fatal Python error: Segmentation fault` in `partition_fx_graph_for_cpu_fallback` → `UnsupportedNodesCollector.run_node` (dynamo_bridge.py:762).

## Root cause
Two bugs stacked:

1. **Tier A (fixed): histc on integer tensor** — `grouped_mm_experts_forward` calls `torch.histc` with an integer expert-index tensor. `TorchFunctionOverride.__torch_function__` passes it through to `histogram_cpu` which does not support integer dtypes. Fixed by adding an int→float cast guarded by a `"histc" in func.__name__` substring check (covers both eager `"histc"` and FX-interpreter `"histc.default"` paths).

2. **Tier B (unfixed): segfault in partition_fx_graph_for_cpu_fallback** — After the histc fix, the FX interpreter in `partition_fx_graph_for_cpu_fallback` runs each op on real TT tensors to classify it as TT-native or CPU-fallback. One of the subsequent ops in `grouped_mm_experts_forward` — most likely `torch.nn.functional.grouped_mm` / `torch._grouped_mm` (the grouped matrix multiplication op used to compute expert outputs after histc) — crashes the TT runtime with a hard segfault rather than raising a Python exception. Because `UnsupportedNodesCollector.run_node` has no try/except around native TT calls, the crash is unrecoverable.

Note: `glm4_moe_lite` does NOT use complex RoPE (`torch.polar` / `view_as_complex` are absent from the model code), so this is a different crash from the `pjrt-complex-tensor-zero-dim-cpu-fallback` seen in GLM-4.7 GGUF models.

## Fix
**Applied (Tier A):** `tt-xla/python_package/tt_torch/torch_overrides.py` — added int→float cast for `histc` in `TorchFunctionOverride.__torch_function__`, using substring match `"histc" in getattr(func, "__name__", "")` to handle both eager and FX-interpreter call paths. Committed to `remediation/huihui_glm_4_7_flash_abliterated-causal_lm-pytorch-4.7_Flash_abliterated-single_device-inference` in tt-xla.

**Proposed (Tier B):** Add exception-safe wrapping to `UnsupportedNodesCollector.run_node` in `torch_xla/_dynamo/dynamo_bridge.py` so that hard crashes from unsupported ops (e.g. `grouped_mm` on TT tensors) are caught and the node is marked as CPU-fallback rather than crashing the process. Alternatively, implement `grouped_mm` support in TT-MLIR/TT-Metal so the op runs natively without falling through to the TT runtime in an unsupported state.

## Tier B justification
**Indicator:** `internal-error-unknown-mechanism` — The specific op that triggers the segfault in `partition_fx_graph_for_cpu_fallback` is unknown without deeper diagnosis (the crash leaves no Python traceback for the faulting op). Additionally, the fix requires either (a) exception-safe infrastructure in the FX partitioner or (b) implementing `grouped_mm` end-to-end in TT-MLIR + TT-Metal, both of which are new-infrastructure / cross-repo changes.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    390.32s (original run), ~167s (post-fix run before segfault)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — histc int→float cast with substring match

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ab03ac0fcb3ebcc1b2662d732b25da691050b32d |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
