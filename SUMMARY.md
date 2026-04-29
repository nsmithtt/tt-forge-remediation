# Remediation Summary: darkidol_llama_3_1_8b_instruct_gguf-causal_lm-pytorch-8B_Instruct_1.2_Uncensored_LWDCLS_IQ_Imatrix_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkidol_llama_3_1_8b_instruct_gguf/causal_lm/pytorch-8B_Instruct_1.2_Uncensored_LWDCLS_IQ_Imatrix_GGUF-single_device-inference]

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

(The original CI failure message was `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` which is raised inside `load_gguf_checkpoint` when gguf is not installed; the actual failure observed post-install is the TypeError above.)

## Root cause
During pytest collection, `TorchDynamicLoader.setup_test_discovery` imports every loader module via `importlib.util.spec_from_file_location`. Many qwen3.5/gpt-oss GGUF loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module level with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` function that does not accept additional keyword arguments. In transformers 5.2.0, `modeling_utils.from_pretrained` added a `model_to_load` kwarg to the `load_gguf_checkpoint` call. Since `modeling_utils` imports `load_gguf_checkpoint` via `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` inside the function body (not at module level), it picks up whatever the module attribute is at call time — i.e., the last broken patch left by a collection-time loader import. When darkidol's `load_model()` is called, it invokes `AutoModelForCausalLM.from_pretrained(...)` which triggers `load_gguf_checkpoint(..., model_to_load=dummy_model)`, hitting the broken patched version and raising TypeError.

26 loaders in tt_forge_models had `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` without `**kwargs`.

## Fix
Added `**kwargs` to the `_patched_load_gguf_checkpoint` signature and forwarded it to `_orig_load_gguf_checkpoint` in all 26 affected loaders in tt_forge_models.

Remediation branch: `remediation/darkidol_llama_3_1_8b_instruct_gguf-causal_lm-pytorch-8B_Instruct_1.2_Uncensored_LWDCLS_IQ_Imatrix_GGUF-single_device-inference` in `tenstorrent/tt-forge-models`

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
- Duration:    484.85s (0:08:04)
- Tier A attempts: N/A

## Files changed
- (26 files in tt_forge_models — see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6a6e2c59c03a8d6576d30e27314de10c2d88315b |
| tt-forge-models | 2a5b09fe91f8b2b5951a740650277e379634d6dc |
