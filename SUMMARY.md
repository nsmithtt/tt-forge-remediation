# Remediation Summary: control_8b_v1_1_i1_gguf-causal_lm-pytorch-Control-8B-V1.1-i1-Q4_K_M-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[control_8b_v1_1_i1_gguf/causal_lm/pytorch-Control-8B-V1.1-i1-Q4_K_M-GGUF-single_device-inference]

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

(The reported failure message "Timeout waiting for ARC msg request queue" was from a different run; the actual failure on branch arch-c-36-tt-xla-dev/nsmith/hf-bringup-11 is the TypeError above, which prevented loading.)

## Root cause
26 GGUF model loaders in tt_forge_models monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
import time with a narrow signature:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
```

transformers 5.2.0 added a `model_to_load` keyword argument to
`load_gguf_checkpoint`. When any of these 26 loaders is imported during
pytest collection, it replaces the function in all transformers module
namespaces with the narrow-signature wrapper. Subsequent calls from
`AutoModelForCausalLM.from_pretrained` that pass `model_to_load=dummy_model`
then raise `TypeError`. The control_8b_v1_1_i1_gguf loader itself does not
patch anything, but it is broken by loaders collected earlier in the session.

## Fix
Changed all 26 GGUF loaders in `tt_forge_models` from:
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

Affected loaders (26 files):
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

Branch: remediation/control_8b_v1_1_i1_gguf-causal_lm-pytorch-Control-8B-V1.1-i1-Q4_K_M-GGUF-single_device-inference
in tenstorrent/tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    437.33s (0:07:17)
- Tier A attempts: N/A

## Files changed
- 26× `tt_forge_models/<model>/causal_lm/pytorch/loader.py` — `_patched_load_gguf_checkpoint` signature widened to `*args, **kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bd0e8180c0b4c8b16664165858fa5eb8c68ca8d7 |
| tt-forge-models | d233dab172a7a9de3bab46d12e15169359bd5829 |
