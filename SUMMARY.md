# Remediation Summary: diver_retriever-embedding_generation-pytorch-Diver-Retriever-4B-1020-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[diver_retriever/embedding_generation/pytorch-Diver-Retriever-4B-1020-single_device-inference]

## Result
FAIL — WH BF16 matmul/fp32-dest-acc precision issue causes PCC=0.864 vs required 0.95; same as tt-metal #39518 / tt-xla #2861

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
tt-metal-bf16-matmul-fp32-dest-acc-precision

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8644872885291851. Required: pcc=0.95.

## Root cause
The Diver-Retriever-4B model (AQ-MedAI/Diver-Retriever-4B-1020) is a 36-layer Qwen3-based embedding model that loads in bfloat16 by default (per its HuggingFace config). On Wormhole hardware, BF16 matmul operations without fp32 destination accumulation enabled produce inaccurate results — a known issue tracked in tt-metal #39518. Through 36 layers of transformer computation, these per-operation errors accumulate to a total PCC of 0.864 against a CPU bfloat16 golden.

The bfloat16 precision floor for this model was measured at CPU-bf16 vs CPU-f32 PCC = 0.956, so even ideal BF16 execution would only reach ~0.956. The TT silicon result of 0.864 is significantly below this floor, confirming a hardware/kernel computation error rather than expected quantization noise. The fp32_dest_acc_en compile option is not set (nullopt), which allows TTNN to use the default BF16 accumulation path that is known to be inaccurate on WH for certain matmul shapes.

An identical precision issue has been observed and classified as Tier B for Gemma 7B (PCC ~0.915 with 28 layers), tracked in tt-xla #2861. This model's deeper network (36 layers vs 28) produces even greater accumulated error (0.864 vs 0.915), consistent with the same root cause. A workaround pattern exists in the loader for mgp_str_base (which references tt-metal #39518 and keeps the model in float32 to avoid triggering BF16 matmul), but a 4B float32 model would require ~16 GB VRAM, exceeding the n150 device's 12 GB capacity, making this workaround infeasible here.

## Fix
No fix applied. The correct fix is in tt-metal: either repair the WH BF16 matmul kernel to accumulate with adequate precision, or ensure fp32_dest_acc_en is automatically enabled for BF16 models on WH devices where BF16 accumulation is inaccurate. Both paths are tracked under tt-metal #39518 and tt-xla #2861. Once the underlying kernel issue is resolved, this model is expected to achieve PCC ≥ 0.956 (the measured BF16 precision floor).

## Tier B justification
cross-cutting — fixing BF16 matmul precision on WH requires either changing the default fp32_dest_acc behavior for all BF16 ops across the compiler pipeline (touching optimizer, lowering passes, and kernel configurations in multiple files/repos) or a hardware-level kernel fix for the WH matmul unit; neither is a scoped single-function change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    140.12s (0:02:20)
- Tier A attempts: N/A

## Files changed
None — no fix applied.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
