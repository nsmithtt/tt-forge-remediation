# Remediation Summary: cotracker3/pytorch-online-single_device-inference

## Skill version
8

## Test
tests/runner/test_models.py::test_all_models_torch[cotracker3/pytorch-online-single_device-inference]

## Result
FAIL — tt-metal runtime L1 overflow in `ttnn::prim::slice` during model execution

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Root exception (from tt-metal):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to
4829696 B which is beyond max L1 size of 1572864 B
```

## Root cause

**Layer: runtime (tt-metal)**

The bringup branch (`ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-range-1500-500-1`) includes a
`_CoTrackerOnlineWrapper` (commit `c935eddd59`) that wraps the `CoTrackerOnlinePredictor` to
present a single-input `forward(video_chunk)` interface suitable for `torch.compile`. The wrapper
performs two sequential predictor calls (once with `is_first_step=True`, once with
`is_first_step=False`) within one compiled graph.

During execution, the compiled slice op (`ttnn::prim::slice`) allocates statically-sized circular
buffers of 4,829,696 B on core (x=0,y=0), which is ~3× the device L1 limit of 1,572,864 B. The
error surfaces as `Bad StatusOr access: INTERNAL: Error code: 13` after tt-metal throws from
`ProgramImpl::validate_circular_buffer_region`.

The backtrace shows the failure path:
```
tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region
tt::tt_metal::distributed::MeshWorkloadImpl::compile
tt::tt_metal::distributed::EnqueueMeshWorkload
ttnn::prim::slice
tt::runtime::ttnn::operations::data_movement::run (SliceOp)
tt::runtime::ttnn::ProgramExecutor::execute
tt::runtime::submit
```

The tensor shapes feeding into slice are large because the online predictor passes full video
feature tensors across its two steps. The resulting intermediate tensors are too wide for the
compiler's current circular buffer sizing strategy.

This is not a loader bug. The wrapper approach is correct — the two-step stateful API must be
collapsed into a single-call form for compilation. The problem is that the compiler backend does
not tile or stream the slice operation, causing it to over-allocate L1 circular buffers.

## Fix
**Proposed fix (tt-mlir / tt-metal layer):**

The slice kernel's circular buffer allocator in `tt_metal/impl/program/program.cpp` (around line
1136) should apply tiling when the required buffer size exceeds the device L1 limit. Alternatively,
tt-mlir's lowering for `stablehlo.slice` should emit a tiled or streaming variant when the operand
tensor's element count exceeds the L1 capacity per core. Both approaches avoid the need for any
loader-side workaround.

No loader change is appropriate — the `_CoTrackerOnlineWrapper` is the correct model-side
adaptation, and reducing input size or grid_size would be a forbidden input-shape workaround.

## Verification
pytest exit status: FAILED (RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13)
Hardware: n150 (Wormhole)
Wall-clock duration: ~400s before exception

## Files changed
None (no fix applied; compiler-stack bug)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | fb4c60411f4a2e50fcf39392c3cbed1f30e9c347 |
