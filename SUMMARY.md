# Remediation Summary: hermes_3_llama_3_1_8b_lorablated_gguf-causal_lm-pytorch-8B_LORABLATED_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hermes_3_llama_3_1_8b_lorablated_gguf/causal_lm/pytorch-8B_LORABLATED_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed transformers 5.2.0 `model_to_load` kwarg breaking change in 26 co-collected GGUF loaders

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

(Original CI report showed `TT_THROW: TIMEOUT: device timeout` on 2026-04-25; by 2026-04-28 the failure manifested as a TypeError at model-load time before device execution — same root cause: transformers 5.2.0 upgrade that changed `load_gguf_checkpoint` to accept `model_to_load`.)

## Root cause
26 loaders in `tt_forge_models` monkey-patch `load_gguf_checkpoint` at module level with a fixed signature `(gguf_path, return_tensors=False)`. During pytest collection all modules are imported, so any of these loaders that is collected before the target test installs the broken patch. Transformers 5.2.0 added a `model_to_load=None` kwarg to `load_gguf_checkpoint` and calls it with that kwarg from `modeling_utils.py:4016` (via a local `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`). Because `modeling_utils.py` imports lazily at call time — not at module level — it picks up the module-attribute patch, hits the broken signature, and raises TypeError. The `hermes_3_llama_3_1_8b_lorablated_gguf` loader itself has no patch; it is a victim of the pollution.

## Fix
In `tt_forge_models`, updated all 26 broken loaders to accept and forward `**kwargs`:

```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```

The 26 affected loaders span three groups:
- Qwen 3.5 architecture patches (`daniloreddy_qwen3_5_*`, `bartowski_coniccat_qwen3_5_*`, `mradermacher_qwen3_5_*`, `mradermacher_bartleby_*`, `mradermacher_crow_*`, `mradermacher_luna_*`, `mradermacher_huihui_*`, `qwen_3_5_imatrix_*`, `tvall43_qwen3_5_*`, `unified_reward_flex_*`, `aaryank_qwen3_5_*`-adjacent variants)
- GPT-OSS Swallow architecture patches (`gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_gpt_oss_swallow_120b_*`, `dmind_3_mini_i1_gguf`)
- Other model patches (`mradermacher_vilm_0_8b_sft_gguf`)

Committed to `tt_forge_models` branch:
`remediation/hermes_3_llama_3_1_8b_lorablated_gguf-causal_lm-pytorch-8B_LORABLATED_GGUF-single_device-inference`

Submodule pointer updated in `tt-xla` branch:
`remediation/hermes_3_llama_3_1_8b_lorablated_gguf-causal_lm-pytorch-8B_LORABLATED_GGUF-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    450.06s (0:07:30)
- Tier A attempts: N/A

## Files changed
26 files in `tt-xla/third_party/tt_forge_models` (one per affected loader):
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4eed8c5d9f78b2d56a6ebe34a881df7e03cbd445 |
| tt-forge-models | 0218b4e7e9cea01fc11e88ba8ea8786b4978e759 |
