# Remediation Summary: mathstral_7b_v0_1_i1_gguf-causal_lm-pytorch-Mathstral_7B_v0_1_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mathstral_7b_v0_1_i1_gguf/causal_lm/pytorch-Mathstral_7B_v0_1_i1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed 26 GGUF loaders missing model_to_load kwarg in _patched_load_gguf_checkpoint

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

(The originally-reported TT_FATAL topology error was not reproduced. The actual failure on the configured branch was the TypeError above.)

## Root cause
26 GGUF loaders (Qwen3.5 family and GPT-OSS-Swallow variants) patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
import time with a wrapper function whose signature is
`_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`.

In transformers 5.x the internal call in `modeling_utils.py:4016` was updated
to pass `model_to_load=dummy_model` as a keyword argument. Because the patch
replaces the module-level attribute, `modeling_utils.from_pretrained` lazily
imports the patched function (via `from .modeling_gguf_pytorch_utils import
load_gguf_checkpoint` inside the method body) and calls it with the new kwarg.
The patched function has no `**kwargs` to absorb it, causing TypeError.

Any test whose alphabetical position comes after one of the 26 affected loaders
will hit this failure if any earlier loader was imported during test collection.

## Fix
In `tt_forge_models`, branch
`remediation/mathstral_7b_v0_1_i1_gguf-gguf-model-to-load-kwarg`, commit
`4c41e61b4fa184c14990433a92882729ed3b2e3e`:

Changed the signature of `_patched_load_gguf_checkpoint` in all 26 affected
loaders from:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

to:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```

Files changed (26 loaders):
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
- Hardware:    wormhole
- Duration:    314.71s (0:05:14)
- Tier A attempts: N/A

## Files changed
- 26 loader files in tt_forge_models (listed above)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4c41e61b4fa184c14990433a92882729ed3b2e3e |
