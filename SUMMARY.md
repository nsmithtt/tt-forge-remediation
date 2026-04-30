# Remediation Summary: hunyuan3d-pytorch-DiT_v2_1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan3d/pytorch-DiT_v2_1-single_device-inference]

## Result
FAIL — MoE infer hang fixed (loader); residual PCC=0.978 < 0.99 is ttmlir-bf16-matmul-precision-floor (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: Test exceeded configured timeout and was killed

After loader fix: AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9780544784099898. Required: pcc=0.99.

## Root cause
Two distinct issues:

1. **Loader bug (fixed):** `MoEBlock.moe_infer` (in `hy3dshape/models/denoisers/moe_layers.py` from the Hunyuan3D-2.1 repo) calls `flat_expert_indices.bincount().cpu().numpy().cumsum(0)`. The `.cpu()` call on a TT device tensor forces a PJRT device-to-host transfer that hangs the TT device, causing the CI timeout.

2. **Compiler bug (Tier B):** After fixing the hang, the model produces PCC=0.978 on n150 WH silicon versus a CPU FP32 reference. CPU BF16 vs FP32 gives PCC=0.9996, confirming the gap is not from BF16 number-format error but from Wormhole BF16 matmul reduced-precision accumulation over 21 transformer layers (hidden_size=2048, FFN inner_dim=8192). This is the same `ttmlir-bf16-matmul-precision-floor` issue seen in Gemma 7B (0.915), Qwen3 4B (0.864), and GPT-J 6B (0.75).

## Fix
**Loader fix (committed):** In `third_party/tt_forge_models/hunyuan3d/pytorch/loader.py`, after importing `MoEBlock`, patch `MoEBlock.moe_infer` with a static per-expert masked matmul. For each of the 8 experts, compute a binary mask over all `num_tokens * moe_top_k` routing assignments, zero out tokens not routed to that expert, and accumulate the weighted expert output. The Python int loop over `range(8)` lets dynamo unroll into 8 static F.linear subgraphs with no dynamic shapes or D2H transfers. Verified numerically equivalent on CPU (max diff 5.96e-8).

**Proposed fix for PCC (Tier B):** Preserve FP32 accumulation through TTIR/TTNN BF16 matmul lowering passes, or force matmul tiles to accumulate in F32 on Wormhole. This is a cross-cutting change across tt-mlir matmul lowering.

## Tier B justification
The `ttmlir-bf16-matmul-precision-floor` bug requires cross-cutting changes to the matmul precision path in tt-mlir/tt-metal (affecting all BF16 models), or enabling F32 accumulation in Wormhole matmul kernels. This touches more than 3 files and requires coordinated changes across the compiler and runtime layers.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    522.27s (0:08:42)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/hunyuan3d/pytorch/loader.py` — static per-expert MoE dispatch

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8af45754bb838520bb9bce1b883409f555acd9b0 |
| tt-forge-models | 91f37920073e740569e21f742965c2c52a99fcd8 |
