# Remediation Summary: l3_15b_mythicalmaid_i1_gguf-causal_lm-pytorch-L3_15B_MYTHICALMAID_T0_0001_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[l3_15b_mythicalmaid_i1_gguf/causal_lm/pytorch-L3_15B_MYTHICALMAID_T0_0001_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — PCC=0.9875 < 0.99 required; residual Tier B ttmlir-bf16-matmul-precision-floor after loader fix

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
Original CI error: `raise InvalidVersion(f"Invalid version: {version!r}")`

Reproduced locally as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After loader fix, test runs to completion with:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9875549522093707. Required: pcc=0.99.
```

## Root cause
**Loader bug (fixed):** Multiple Qwen3.5 GGUF loaders (26 files) monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import
time with a narrow-signature wrapper `(gguf_path, return_tensors=False)`. Since
all loaders are imported at pytest collection time, this patched function is
active when the l3_15b test runs. `transformers 5.2.0` added `model_to_load` as
a kwarg in `from_pretrained`, which calls `load_gguf_checkpoint` via a late
import (`from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` inside
the function body), picking up the patched version and raising `TypeError`.

**Residual (Tier B):** After the loader fix, the 15B LLaMA-3 model compiles and
runs on n150 but produces PCC=0.9875 against CPU reference. The gap (0.0125) is
consistent with the known WH BF16 matmul precision floor that compounds over
the model's 48 hidden layers, the same class as BlackSheep 12B (PCC=0.949),
Qwen3 4B (PCC=0.864), and Gemma 7B (PCC=0.915). Cross-cutting fix required.

## Fix
**Loader (applied):** Updated all 26 GGUF loaders in `tt_forge_models` that
define `_patched_load_gguf_checkpoint` with the narrow signature
`(gguf_path, return_tensors=False)` to use `(*args, **kwargs)` and pass
them through to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

Files changed: `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`,
`unified_reward_flex_qwen35_27b_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`,
`mradermacher_qwen3_5_4b_unfiltered_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`,
`mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_unredacted_max_gguf`,
`mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`,
`mradermacher_vilm_0_8b_sft_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`,
`mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`,
`mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`,
`mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_bartleby_qwen3_5_4b_gguf`,
`mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`,
`gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`,
`qwen_3_5_imatrix_gguf`, `dmind_3_mini_i1_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`,
`bartowski_coniccat_qwen3_5_27b_writer_gguf` — all in `tt_forge_models/<model>/causal_lm/pytorch/loader.py`

**Residual (not applied):** BF16 matmul precision floor requires cross-cutting
changes in `tt-mlir` to preserve F32 accumulation across all lowering passes.

## Tier B justification
Cross-cutting: the WH BF16 matmul precision floor affects every matmul in the
model's 48 hidden layers; fixing it requires preserving F32 precision across
all lowering passes in `tt-mlir`, touching more than 3 files across multiple
passes.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    674.10s (0:11:14)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/*/causal_lm/pytorch/loader.py` (26 files) — `(*args, **kwargs)` fix
- `tt-xla/third_party/tt_forge_models` — submodule pointer updated

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4c1126758a80db02893f6cdf6ecd7698c53b2cfb |
| tt-forge-models | 3b16100ccfa8fa482764bed27afcb7a2f474e3e5 |
