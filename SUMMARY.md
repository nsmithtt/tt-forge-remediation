# Remediation Summary: animatediff_motion_lora_pan_right/pytorch-PanRight-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[animatediff_motion_lora_pan_right/pytorch-PanRight-single_device-inference]

## Result
FAIL — sdpa_decode TT_FATAL with seq_len=1 from temporal attention with num_frames=1

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Two layered issues:

**Issue 1 (loader, fixed):** After `load_lora_weights()`, the PEFT LoRA wrapper
(`diffusers/utils/peft_utils.py`) installs a decorated forward on the UNet. This
wrapper appeared in the call stack at test-failure time. Calling
`fuse_lora()` merges LoRA delta weights into the base weights and is a standard
deployment step (output is mathematically equivalent).

**Issue 2 (runtime, blocking):** The AnimateDiff temporal attention block
(`AnimateDiffTransformer3D`) with `num_frames=1` creates a scaled-dot-product
attention call with Q/K/V of sequence length 1. TTNN dispatches this to the
`sdpa_decode` kernel. In
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp` (lines
63–66), the kernel calls:

```cpp
TT_FATAL(
    k_chunk_size % 32 == 0,
    "Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: {}",
    k_chunk_size);
```

`get_chunk_size(1)` returns 1 (max power-of-2 divisor of 1), so `1 % 32 != 0`
and the kernel aborts. The abort propagates through the TT runtime as a failed
StatusOr, raising `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

Changing `num_frames` to ≥ 32 to avoid the `sdpa_decode` dispatch is explicitly
a forbidden workaround ("Do not raise num_frames … to dodge constraints like
SDPA k_chunk_size >= 32"). The fix must be in the runtime layer.

## Fix

**Applied (loader layer):**
- `fuse_lora()` added to `load_model()` — merges LoRA delta into base weights,
  removes the PEFT wrapper from the compiled graph.
- Default dtype changed to `torch.bfloat16` — halves activation memory vs
  float32, consistent with all other AnimateDiff LoRA loaders.
- `encoder_hidden_states` batch corrected to `batch_size * num_frames` — matches
  the internal reshape `(B, C, T, H, W) → (B*T, C, H, W)` that UNetMotionModel
  performs before cross-attention.

**Proposed fix (runtime layer — tt-metal):**
In `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp`, handle
the case where `k_chunk_size < 32` by either:
1. Clamping `k_chunk_size` to `min(get_chunk_size(s), 32)` and padding the KV
   sequence to a multiple of 32 (with masking), or
2. Falling back to the regular (non-decode) SDPA kernel when `s < 32`.

This affects any model that runs temporal attention with fewer than 32 frames,
e.g. single-frame inference with AnimateDiff.

## Verification
pytest exit status: FAILED, 1 failed in 414.60s — TT silicon (p150b / Blackhole)

## Files changed
- `animatediff_motion_lora_pan_right/pytorch/loader.py` (in tt-forge-models,
  branch `remediation/animatediff-motion-lora-pan-right`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0fccc0cbb6670491f450bbc7b90472ad789bdba4 |
| tt-forge-models | 7d3dd0fcb3c21b1b244e842f7cb3b17dd40666ff |
