# Remediation Summary: deepseek_r1_distill_qwen_7b_heretic_i1_gguf-causal_lm-pytorch-DeepSeek_R1_Distill_Qwen_7B_Heretic_i1_Q4_K_M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_7b_heretic_i1_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Qwen_7B_Heretic_i1_Q4_K_M-single_device-inference]

## Result
FAIL — Loader TypeError fixed; PCC=0.9441 remains below required 0.99; root cause is WH BF16 matmul precision accumulation across 28 layers (Tier B)

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
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9441219438622457. Required: pcc=0.95.

During reproduction we hit a second blocker first:
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause

**Loader bug (fixed):** During test collection, `setup_test_discovery` eagerly imports all model loaders.
Twenty-five GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature that does not accept
the `model_to_load` kwarg added in transformers 5.2.0. The last such loader alphabetically
(`unified_reward_flex_qwen35_27b_gguf`) leaves the broken version in place. When the DeepSeek test
runs, `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`
and hits the TypeError.

**Compiler-stack bug (unfixed, Tier B):** After fixing the TypeError, the model (Qwen2 architecture,
28 layers, hidden_size=3584) loads and compiles successfully. However, the TT output PCC compared to
CPU BF16 reference is 0.9441, well below both the default threshold (0.99) and the original threshold
(0.95). This is the same WH silicon BF16 matmul accumulation precision issue observed for Gemma 7B
(PCC~0.915, 32 layers) and Qwen3 4B (PCC=0.864, 36 layers). The precision gap grows with model depth:
28 layers gives 0.9441, consistent with per-layer BF16 matmul error accumulation.

## Fix

**Loader fix applied** in `tt_forge_models` on branch
`remediation/deepseek_r1_distill_qwen_7b_heretic_i1_gguf-causal_lm-pytorch-single_device-inference`:

Cherry-picked commit `8c52e71618cd5b4443b9cb22b8d0406237ca14af` ("Fix load_gguf_checkpoint patchers:
accept **kwargs for model_to_load compat") which updates all 25 broken GGUF loaders from:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

to:

```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

**Compiler-stack fix (proposed, not attempted):** The WH BF16 matmul precision accumulation should be
addressed in tt-mlir by preserving higher-precision accumulation through the StableHLO→TTIR lowering
passes. This is tracked as tt-xla #2861.

## Tier B justification

Which indicator: `cross-cutting`

The BF16 matmul precision issue affects all matmul operations across all lowering passes for all models.
Preserving f32 accumulation through the entire compilation pipeline requires coordinated changes across
multiple passes in tt-mlir and changes to how tt-metal executes matmul kernels. This is not a scoped
single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    367.44s (0:06:07)
- Tier A attempts: N/A

## Files changed
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

(All in `tt_forge_models` submodule — no changes to tt-mlir, tt-metal, or tt-xla.)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3aa60f64cf444b5e2d421f298c8374a01b2aabba |
| tt-forge-models | 8c52e71618cd5b4443b9cb22b8d0406237ca14af |
