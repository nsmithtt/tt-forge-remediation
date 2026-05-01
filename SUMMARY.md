# Remediation Summary: moose_star_ir_r1d_7b_i1_gguf-causal_lm-pytorch-MOOSE_Star_IR_R1D_7B_i1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moose_star_ir_r1d_7b_i1_gguf/causal_lm/pytorch-MOOSE_Star_IR_R1D_7B_i1_Q4_K_M-single_device-inference]

## Result
FAIL — ttmlir-bf16-matmul-precision-floor: PCC=0.949 on BH p150b for 28-layer Qwen2 7B model; below both 0.99 (default) and 0.95 (original CI threshold)

## Stack layer
tt-mlir

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9490561302719787. Required: pcc=0.95.

## Root cause
Two bugs found in sequence:

**Bug 1 (loader, fixed):** 26 GGUF loaders (`unified_reward_flex_qwen35_27b_gguf`,
`tvall43_qwen3_5_*`, `mradermacher_qwen3_5_*`, `gpt_oss_swallow_*`, etc.) defined
`_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — a narrow signature
that clobbered the global `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
during test collection. When the MOOSE test then ran, `AutoModelForCausalLM.from_pretrained`
called the clobbered function with the new `model_to_load=dummy_model` kwarg introduced
in transformers 5.2.0, producing `TypeError: _patched_load_gguf_checkpoint() got an
unexpected keyword argument 'model_to_load'`.

**Bug 2 (tt-mlir, Tier B):** After the loader fix, the model loaded and ran. The Qwen2
28-layer (hidden_size=3584) i1-Q4_K_M GGUF on Blackhole p150b gives PCC=0.949 vs CPU
BF16 golden. This matches the known ttmlir-bf16-matmul-precision-floor pattern seen in
other multi-layer models on BH: BlackSheep-RP 12B (Q4_K_M, BH p150b) also yielded
PCC=0.949. The error accumulates across 28 matmul layers due to WH/BH BF16 accumulation
precision differing from CPU BF16 FP32-accumulate.

## Fix
**Bug 1 (applied):** All 26 narrow-sig patches updated to `(*args, **kwargs)` with
pass-through to `_orig_load_gguf_checkpoint(*args, **kwargs)` so any new transformers
kwargs are forwarded transparently.

Files changed in `tt_forge_models` (26 loaders):
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

**Bug 2 (unfixed, Tier B):** The BF16 matmul precision floor requires cross-cutting
changes to accumulate in FP32 through every matmul lowering pass in tt-mlir, which is
new infrastructure affecting all BF16 models.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The BF16 matmul precision floor affects all 28 matmul layers in the Qwen2 forward pass.
Fixing it requires preserving FP32 accumulation through every matmul lowering in tt-mlir
(TTIR→TTNN path), a cross-cutting change across multiple passes and files that goes
beyond a single scoped fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    435.19s (0:07:15) after loader fix
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 loader files — widened `_patched_load_gguf_checkpoint` signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d99d0d66c61101fbb442a8170b5424e14fbb0f9a |
| tt-forge-models | 67677ea7b14d5f58c7af801825c032af49aa12b8 |
