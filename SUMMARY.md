# Remediation Summary: davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf-causal_lm-pytorch-Qwen2_5_MOE_2X1_5B_DeepSeek_Uncensored_Censored_4B_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf/causal_lm/pytorch-Qwen2_5_MOE_2X1_5B_DeepSeek_Uncensored_Censored_4B_Q4_K_M-single_device-inference]

## Result
FAIL — PCC=0.8929 vs required=0.99; WH BF16 matmul precision floor after all loader bugs fixed

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-moe

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: RuntimeError: Check failed: status.ok(): MHLO -> StableHLO conversion failed.

After loader fixes, final failure:
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8929454085030116. Required: pcc=0.99.

## Root cause

Five cascading loader bugs prevented the model from reaching the compiler at all.
After all loader fixes, the model compiles and runs on silicon, but TT silicon's
BF16 matmul accumulation introduces significant precision loss:
- CPU BF16 vs FP32: PCC = 0.9858 (BF16 floor only 0.014 below FP32)
- TT vs CPU FP32: PCC = 0.8929 (0.093 below FP32)

The gap (0.093) is TT-specific WH BF16 matmul accumulation error, not inherent
BF16 quantization noise. This is the same issue documented for Qwen3-4B (0.864)
and Gemma-7B (~0.915).

The loader bugs fixed were:

1. **model_to_load TypeError** — 26 GGUF loaders across tt_forge_models patched
   `load_gguf_checkpoint` with signature `(gguf_path, return_tensors=False)`.
   Transformers 5.2.0 added `model_to_load` kwarg. When test discovery imported
   all loaders, the broken patch remained on the global function and caused
   `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument
   'model_to_load'`. Fixed by advancing tt_forge_models to commit 8051c4095b.

2. **GGUF metadata moe_intermediate_size mismatch** — The GGUF metadata KV
   incorrectly inherits Qwen2-MoE-57B-A14B values (`moe_intermediate_size=1408`,
   `shared_expert_intermediate_size=5632`). The actual tensors
   (`blk.0.ffn_gate_exps.weight` shape `[1536, 8960, 2]`) have
   `moe_intermediate_size=8960` for both expert and shared-expert projections.
   Fixed by reading actual GGUF tensor shapes via `GGUFReader` and overriding
   the config before `AutoModelForCausalLM.from_pretrained`.

3. **Expert dispatch segfault** — `Qwen2MoeExperts.forward` default path uses
   `nonzero()`/`torch.where()` for expert dispatch, which crashes the XLA dynamo
   bridge during graph partitioning (`partition_fx_graph_for_cpu_fallback`).
   Fixed by replacing the expert forward with a static per-expert masked matmul.

4. **EmbeddingsDeviceOperation L1 overflow** — The `batched_mm` expert path does
   `self.gate_up_proj[expert_ids_clamped]` which is lowered to an `EmbeddingOp`
   with "row size" = `17920 * 1536 * 2 = 55 MB`, overflowing the 1.5 MB L1
   limit. Fixed by the same static per-expert loop (Fix 3), which uses
   `gate_up_proj[int_idx]` (plain 2-D slice, not an embedding lookup).

5. **Float32 dtype propagation** — `Qwen2MoeTopKRouter` computes routing weights
   in `float32` (`softmax(dtype=float)`). In the static expert loop, `expert_out
   * routing_weight` upcasts to float32 and propagates through the residual
   connection. The next layer's `q_proj` (BF16 weights) then receives float32
   input, causing `RuntimeError: mat1 and mat2 must have the same dtype, but got
   Float and BFloat16`. Fixed by casting `top_k_weights` to `hidden_states.dtype`
   at the entry of the static forward.

The remaining PCC failure (0.8929) is not from the loader fixes: the model has
`num_experts=2`, `num_experts_per_tok=2` (all tokens select all experts), so the
static dense computation is mathematically identical to the original sparse
dispatch. The precision loss is purely from TT silicon's BF16 matmul
accumulation over 28 layers with `moe_intermediate_size=8960`.

## Fix
All five loader bugs fixed in tt_forge_models at:
  `third_party/tt_forge_models/davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf/causal_lm/pytorch/loader.py`

Remediation branch in tt_forge_models:
  `remediation/davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf`
  commits: f3a3787e7f, aa391c8728, 528b16f0da, 26c324aa12

tt-xla remediation branch:
  `remediation/davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf-causal_lm-pytorch-Qwen2_5_MOE_2X1_5B_DeepSeek_Uncensored_Censored_4B_Q4_K_M-single_device-inference`

The PCC failure requires compiler-stack work: higher-precision matmul accumulation
on WH silicon, or selective F32 upcast for MoE projections in tt-mlir.

## Tier B justification
cross-cutting — fixing WH BF16 matmul precision requires changing accumulation
precision across all matmul lowerings or selectively promoting MoE projections to
F32, touching multiple files in tt-mlir and affecting all models using BF16.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    463.97s (0:07:43)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/davidau_qwen2_5_moe_2x1_5b_deepseek_uncensored_censored_4b_gguf/causal_lm/pytorch/requirements.txt` (added in prior commit)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 70b0e0daaaae1756aa12cf8a4848e19114b53532 |
| tt-forge-models | 26c324aa12a95e5225ce6a18fa3c59c69b0c17fc |
