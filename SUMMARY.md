# Remediation Summary: animatediff-motion-lora-tilt-up-pytorch-TiltUp-single-device-inference

## Skill version
13

## Test
tests/runner/test_models.py::test_all_models_torch[animatediff_motion_lora_tilt_up/pytorch-TiltUp-single_device-inference]

## Result
FAIL — Conv2d sharded program factory CB page size underestimated in tt-metal for 8×8 latent spatial dims

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Root assertion in tt-metal:
```
TT_FATAL @ tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp:665:
access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size >= conv_reader_indices_storage.get_buffer()->page_size()
info: CB page size 64 should be greater than the config tensor page size 68
```

## Root cause
**Layer: tt-metal (runtime — conv2d kernel configuration)**

Two bugs were present:

1. **Loader bug (fixed):** `load_model()` returned the full `AnimateDiffPipeline` (not a `torch.nn.Module`) and `load_inputs()` returned `{"prompt": str}` instead of UNet tensor inputs. These caused an `AssertionError` in the test infra before the compiler was even reached.

2. **Compiler-stack bug (unfixed):** After the loader was fixed to return `self.pipeline.unet` (a `UNetMotionModel`) with proper tensor inputs — `sample: (1, 4, 16, 8, 8)`, `encoder_hidden_states: (16, 77, 768)` — the test fails during compilation/execution on TT hardware.

   The failure is inside `Conv2dShardedProgramFactory::create` in tt-metal. When `config_tensors_in_dram` is true, the code asserts that the pre-calculated CB READER_INDICES page size (computed by `get_cb_info`) is ≥ the actual config tensor page size (computed by `construct_on_host_config_tensor`). For the `conv_in` layer of the `UNetMotionModel` operating on `(16, 4, 8, 8)` input, `get_cb_info` returns a READER_INDICES page size of **64 bytes**, but the actual config tensor requires **68 bytes** — 4 bytes over the pre-allocated CB size.

   The comment at that site says "The actual CB reader size is difficult to calculate in calculate_L1_size. So instead keep the CB size as the maximum possible size." — meaning the ceiling calculation is incorrect for this input shape.

## Fix
**Loader fix (applied, in `tt_forge_models`):**
- `load_model()` now returns `self.pipeline.unet` after calling `self.pipeline.fuse_lora()`.
- `load_inputs()` now returns tensor inputs: `sample (1,4,16,8,8)`, `timestep (1,)`, `encoder_hidden_states (16,77,768)`.
- Added `unpack_forward_output()` to extract `.sample` from `UNetMotionOutput`.
- None of these changes are forbidden workarounds — returning the UNet directly is the correct pattern (matching the base `animatediff` loader), and the tensor shapes are the natural AnimateDiff defaults (16 frames, 8×8 latent).

**Compiler-stack bug (not fixed — requires tt-metal work):**

The proposed fix lives in `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp` around line 665, and/or in the `get_cb_info()` function in `conv2d_op_program_factory_common.cpp`.

The READER_INDICES CB page size calculation must account for the exact byte count produced by `construct_on_host_config_tensor` for small spatial inputs (H=8, W=8). The fix is to either:
- Correct the formula in `get_cb_info` to accurately compute the maximum reader indices page size for all `(H, W)` combinations, or
- Add 32-byte alignment padding to ensure the pre-calculated ceiling is always ≥ the actual tensor's page size.

## Verification
pytest exited FAILED — hardware: n150 — wall-clock: ~339s to compile then assert

## Files changed
- `tt_forge_models/animatediff_motion_lora_tilt_up/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 882afbb0f2a0594de56770fed8fa53c9929a6743 |
| tt-forge-models | 5a6b4e08bd74016b3281f13ec5a4fe5f80643d65 |
