# Remediation Summary: general_chat_llama_3_2_3b_dpo_gguf-causal_lm-pytorch-GeneralChat-Llama3.2-3B-DPO-GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[general_chat_llama_3_2_3b_dpo_gguf/causal_lm/pytorch-GeneralChat-Llama3.2-3B-DPO-GGUF-single_device-inference]

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Originally reported as `AttributeError: 'NoneType' object has no attribute 'config'`; the TypeError is the proximate failure when collection imports GGUF loaders before the target test runs.)

## Root cause
26 GGUF loader files in `tt_forge_models` define a `_patched_load_gguf_checkpoint` that monkey-patches `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` (and three other module attributes) at import time. Their patched signature is `(gguf_path, return_tensors=False)`, which is missing the `model_to_load` keyword argument added in transformers 5.x. During pytest collection all loader modules are imported, so by the time the target test runs the global `load_gguf_checkpoint` has been replaced by the final link in the 26-patcher chain. When `AutoModelForCausalLM.from_pretrained` does a local import:

```python
from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
state_dict = load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)
```

it picks up the patched version, which rejects `model_to_load` with `TypeError`. This is a loader-layer bug: the patchers are in `tt_forge_models`, not in any compiler component.

## Fix
In `tt_forge_models`, on branch `remediation/general_chat_llama_3_2_3b_dpo_gguf-causal_lm-pytorch-GeneralChat-Llama3.2-3B-DPO-GGUF-single_device-inference`, commit `14c09a32b4`:

Two-line change per file across all 26 affected loaders:

1. Add `model_to_load=None` to the patched signature:
   ```python
   # before
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
   # after
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
   ```

2. Forward `model_to_load` through to the wrapped original:
   ```python
   # before
   result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
   # after
   result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
   ```

Applied via `sed` to all 26 files with the broken pattern. Files already using `*args, **kwargs` (e.g. `gpt_oss_swallow_20b_sft_v0_1_gguf`) were unaffected.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    370.51s (0:06:10)
- Tier A attempts: N/A

## Files changed
26 loader files in `tt_forge_models`:
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
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 14c09a32b4915f96027841146e889b8976fda3a0 |
