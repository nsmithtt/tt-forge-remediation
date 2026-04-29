# Remediation Summary: bartowski_jackrong_qwen3_5_4b_neo_gguf-causal_lm-pytorch-Jackrong_Qwen3.5-4B-Neo-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_jackrong_qwen3_5_4b_neo_gguf/causal_lm/pytorch-Jackrong_Qwen3.5-4B-Neo-GGUF-single_device-inference]

## Result
FAIL — GatedDeltaNet recurrent kernel produces PCC 0.325–0.688 on TT silicon vs CPU; root cause is float32 precision not preserved through TT lowering passes

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `raise RuntimeError(` (weight shape mismatch — GGUF architecture "qwen35" not
recognized, model loaded as `Qwen3ForCausalLM` instead of `Qwen3_5ForCausalLM`, so
q_proj expected [2048,2560] but checkpoint had [8192,2560]).

After loader fix: `AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.68814325544458. Required: pcc=0.99.`

Independent diagnostic (loader-only, no pytest framework) measured PCC=0.3255 between
TT-device output and CPU golden.

## Root cause
Two separate issues were encountered and fixed in sequence:

**Issue 1 (loader bug — fixed):** The GGUF architecture string "qwen35" had no config-field
or tensor-name mapping for `Qwen3_5ForCausalLM`. The existing bartowski_coniccat loader
(alphabetically earlier, imported first at collection time) was mapping "qwen35" → "qwen3",
causing the model to load as `Qwen3ForCausalLM` with wrong head_dim=128 instead of 256.
This produced the original weight mismatch RuntimeError.

**Issue 2 (compiler-stack bug — Tier B, unfixed):** After the loader fix, the model loads
correctly as `Qwen3_5ForCausalLM` and compiles on TT silicon without error, but produces
PCC 0.325–0.688 instead of ≥0.99.

Root cause of Issue 2: `Qwen3_5GatedDeltaNet.forward` (the linear-attention layers present
in 28 of 32 model layers) contains explicit `.float()` casts at numerically sensitive
points (e.g. `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)` at
`modeling_qwen3_5.py:586`). The `torch_chunk_gated_delta_rule` fallback kernel then
mixes float32 `g` with bfloat16 query/key/value tensors; PyTorch upcasts to float32, but
TT does not preserve float32 through its lowering passes. The systematic precision
degradation compounds across 28 GatedDeltaNet layers, yielding PCC ≈ 0.33–0.69.

## Fix
**Loader fixes (three commits in tt_forge_models remediation branch):**

1. `a201bd9930` — Added full `_QWEN35_CONFIG_MAPPING` (SSM fields: `ssm.conv_kernel`,
   `ssm.state_size`, `ssm.group_count`, `ssm.time_step_rank`, `full_attention_interval`),
   `_Qwen35TensorProcessor` for SSM weight transforms (A_log sign, conv1d reshape),
   `perform_fallback_tensor_mapping` for dt_bias, `_patched_get_gguf_hf_weights_map`
   translating `qwen3_5_text → qwen35` for gguf-py arch lookup, and
   `_patched_load_gguf_checkpoint` setting `model_type = "qwen3_5_text"`.

2. `0196a6f61a` — Changed Neo detection from `model_type == "qwen35"` to
   `"full_attention_interval" in config`: the coniccat loader (imported before jackrong
   alphabetically) already translated "qwen35" → "qwen3", so the model_type check failed.
   `full_attention_interval` is present only in Neo hybrid GGUF files and survives the
   coniccat patch.

3. `b6a8037b18` — Set `model_kwargs = {"use_cache": False}` in `load_model` to prevent
   `Qwen3_5DynamicCache` from appearing in model output, which caused
   `torch.equal(x, Qwen3_5DynamicCache)` TypeError in the comparison evaluator.

**Compiler-stack fix:** None attempted (Tier B).

**Proposed fix:** Ensure TT preserves float32 semantics for ops that follow an explicit
`.float()` cast in the StableHLO graph, or implement f32 accumulation paths in the TTNN
reduction and elementwise kernels used by GatedDeltaNet. This would require cross-cutting
changes to tt-mlir's lowering passes and is Tier B.

## Tier B justification
Indicator: **cross-cutting**

Preserving f32 precision through every lowering pass in tt-mlir is explicitly a cross-
cutting change. The GatedDeltaNet kernel uses explicit `.float()` at multiple points
(gate computation, decay mask, recurrent state casts) across 28 layers. Fixing this
requires ensuring f32 precision is maintained through elementwise, matmul, cumsum, exp,
and softplus ops in the TTNN backend — changes spanning multiple lowering files and
potentially the hardware kernel level.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1857.46s (0:30:57) for 5th run (use_cache fix); PCC 0.688
- Independent diagnostic: PCC 0.3255 (TT bfloat16 vs CPU bfloat16, same loader)
- Tier A attempts: N/A (Tier B — no fix attempted in compiler stack)

## Files changed
- `tt-xla/third_party/tt_forge_models/bartowski_jackrong_qwen3_5_4b_neo_gguf/causal_lm/pytorch/loader.py`
  (new file — full loader implementation with qwen35 GGUF support)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 223d2fdd758c6b276eb80b6cffc359733d600327 |
| tt-forge-models | b6a8037b18dd8eb1bf01e97050de5d5c016eeb90 |
