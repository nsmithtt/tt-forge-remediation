# Remediation Summary: ggml_org_qwen3_14b_gguf-causal_lm-pytorch-14B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ggml_org_qwen3_14b_gguf/causal_lm/pytorch-14B_GGUF-single_device-inference]

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

(The initial failure in CI was recorded as `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` — gguf 0.18.0 is present in the test environment, so the actual reproduction failure was the TypeError.)

## Root cause
26 GGUF loaders in tt_forge_models (qwen3.5 and gpt-oss variants) monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import
time with a narrow-signature wrapper:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
```

During pytest collection, `TorchDynamicLoader.setup_test_discovery` imports all
model loader modules. When any of these 26 loaders is imported before the
Qwen3-14B test runs, it replaces the global `load_gguf_checkpoint` attribute on
`transformers.modeling_gguf_pytorch_utils`. Subsequently, `modeling_utils.py`
does a local import of that symbol at line 4010:

```python
from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
```

It calls the patched function with `model_to_load=dummy_model` (added in
transformers 5.2.0), which raises `TypeError` because the narrow-signature
wrapper does not accept that keyword argument. The Qwen3-14B loader itself
has no patch — it is a victim of the cross-loader clobbering.

Additionally, the ggml_org_qwen3_14b_gguf loader was missing a requirements.txt
declaring `gguf>=0.10.0`.

## Fix
Fixed in `tt_forge_models` on branch
`remediation/ggml_org_qwen3_14b_gguf-causal_lm-pytorch-14B_GGUF-single_device-inference`:

1. Changed all 26 narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   definitions to `(*args, **kwargs)`, forwarding all arguments to the original
   function. This matches the pattern used by the two loaders that had already
   been fixed.

2. Added `gguf>=0.10.0` to a new
   `ggml_org_qwen3_14b_gguf/causal_lm/pytorch/requirements.txt`.

The tt-xla remediation branch advances the `third_party/tt_forge_models`
submodule pointer to commit `168b0ea507` on
`remediation/ggml_org_qwen3_14b_gguf-causal_lm-pytorch-14B_GGUF-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    540.92s (0:09:00)
- Tier A attempts: N/A

## Files changed
- `ggml_org_qwen3_14b_gguf/causal_lm/pytorch/requirements.txt` (new file, `gguf>=0.10.0`)
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

`tt-xla` commit `7adb0f194` on branch
`remediation/ggml_org_qwen3_14b_gguf-causal_lm-pytorch-14B_GGUF-single_device-inference`
(advances `third_party/tt_forge_models` pointer to `168b0ea507`)

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | 7adb0f194032901438d4d4c57a98a4ba60489aa9   |
| tt-forge-models | 168b0ea50768107402cf2f3794bf6a39c76da706   |
