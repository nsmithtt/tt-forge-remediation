# Remediation Summary: athena_3_14b_i1_gguf-causal_lm-pytorch-Athena-3-14B-i1-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[athena_3_14b_i1_gguf/causal_lm/pytorch-Athena-3-14B-i1-GGUF-single_device-inference]

## Result
SILICON_PASS

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The CI report showed this as `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` because the TypeError propagated through transformers' gguf loading path.)

## Root cause
26 GGUF loaders in tt_forge_models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a narrow signature:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
```

These loaders are imported during pytest collection. Once any one of them is imported, the global `load_gguf_checkpoint` is replaced with the narrow-signature version. When the athena_3_14b test later calls `AutoModelForCausalLM.from_pretrained(...)`, transformers 5.2.0 internally calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the patched function cannot accept, raising `TypeError`.

The 26 affected loaders are in the `mradermacher_*`, `gpt_oss_swallow_*`, `tvall43_*`, `qwen_3_5_imatrix_gguf`, `dmind_3_mini_i1_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`, and `unified_reward_flex_qwen35_27b_gguf` families. A handful of loaders in the same codebase had already been fixed to use `*args, **kwargs`.

## Fix
In `tt_forge_models` (branch `remediation/athena-3-14b-i1-gguf-causal-lm-pytorch-single-device-inference`, commit `47d477decf`): changed all 26 narrow-signature wrappers from

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

to

```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```

Files changed in tt_forge_models:
- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    684.76s (0:11:24)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 loader.py files (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d809ec896d565f7a3c273296c3b00ba972c161c5 |
| tt-forge-models | 47d477decf1068a4c569794a8549a3b844b4381f |
