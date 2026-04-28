# Remediation Summary: aria/pytorch-25B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aria/pytorch-25B-single_device-inference]

## Result
FAIL — Aria 25B MoE language model hangs on TT hardware (physical core timeout)

## Failure
```
2026-04-17 23:43:08.395 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)
```

## Root cause

Two bugs were found, one in the loader and one in the compiler stack.

**Bug 1 (loader, fixed):** transformers 5.x `@capture_outputs` decorator uses a
`CompileableContextVar` to collect vision encoder hidden states via forward hooks.
When `torch.compile` compiles `AriaModel.forward`, it creates a graph break at the
`get_image_features` call (line 994 of `modeling_aria.py`). The subgraph before
the break runs compiled; `get_image_features` (including the vision tower) runs
eagerly after the break. The `@capture_outputs` wrapper calls
`_active_collector.set(collected_outputs)` via a `contextvars.ContextVar` before
dispatching to the vision tower forward. However, the compiled first subgraph was
already dispatched from a copy of the compilation-time context, meaning the
ContextVar's value set at call-time is not visible inside any compiled subgraph.
The forward hooks in the vision encoder therefore see `_active_collector.get() ==
None` and collect nothing, leaving `hidden_states = ()`. When
`get_image_features` later accesses `image_outputs.hidden_states[-1]`
(vision_feature_layer=-1), it gets `IndexError: tuple index out of range`.
The unhandled Python exception leaves the device in an unrecoverable state,
producing the reported `TIMEOUT`.

**Bug 2 (compiler stack, unfixed):** After applying the loader fix, the vision
tower runs correctly and produces 28 hidden states. The second dynamo subgraph
(resumed at `modeling_aria.py:994`, `torch_dynamo_resume_in_forward_at_994`)
contains the full Aria 25B MoE transformer, including `AriaSparseMixedExpertsMLP`
with its `index_select`-based expert routing
(`AriaGroupedExpertsGemm.forward` at line 330). When this subgraph is dispatched
to the TT device via `extract_compiled_graph → sync`, physical TT cores time out:

```
2026-04-28 01:28:06.087 | critical |          Always | TT_THROW: Device 0: Timeout (10000 ms) waiting for physical cores to finish: (x=13,y=6), (x=11,y=7), ...
```

This is a hang in the TT-MLIR-compiled MoE language model subgraph. The most
likely cause is the MoE sparse expert routing (dynamic `index_select` +
variable-length scatter/gather over expert assignments), which may not be
supported or may produce an infinite loop in tt-metal.

## Fix

**Loader fix (applied):** Patch `output_capturing._active_collector.set` and
`reset` to always use `global_var` instead of the `ContextVar`. This makes the
`_active_collector` value visible inside compiled torch.compile subgraphs while
preserving the `@capture_outputs` graph break, so the vision tower continues to
run eagerly (as intended). This is not a forbidden workaround: the vision tower
still runs fully on device (eagerly, not CPU-offloaded), and no model depth,
inputs, or thresholds were changed.

**Proposed compiler fix:** Investigate and fix `AriaGroupedExpertsGemm` /
`AriaSparseMixedExpertsMLP` execution on tt-metal. Specifically:
- Check whether `aten::index_select` with a dynamically-computed index tensor
  is correctly lowered to StableHLO and then to tt-metal ops.
- The sorted expert dispatch (`permuted_tokens = hidden_states.index_select(0,
  sorted_indices // config.moe_topk)`) produces variable-length sub-tensors per
  expert which may require dynamic shapes support or a different lowering
  strategy.
- Alternatively, check if the MoE router produces an infinite loop in the
  compiled kernel (e.g. loop bounds derived from dynamic routing not properly
  unrolled).

## Verification
Test exited SIGABRT (exit code 134) after ~34 seconds on Wormhole n300 hardware.
The loader fix was verified to correctly populate `hidden_states` (28 states) under
`torch.compile(backend='eager')` before the silicon run. The device timeout
reproduces identically on each run.

## Files changed
- `tt-xla/third_party/tt_forge_models/aria/pytorch/loader.py` — add
  `_patch_capture_outputs_for_compile()` and call it from `load_model()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d1547acdd (pending push — HEAD of local branch) |
| tt-forge-models | ea47035df195cc29615466f9830708233d259f2f |
