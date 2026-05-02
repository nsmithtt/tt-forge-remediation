# Remediation Summary: huihui_qwen_3_8b_abliterated_v2_i1_gguf-causal_lm-pytorch-8B_Abliterated_v2_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch-8B_Abliterated_v2_GGUF-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed: missing gguf>=0.10.0, 26 narrow-sig _patched_load_gguf_checkpoint causing model_to_load TypeError, missing chat_template guard

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
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Three sequential loader bugs in `huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch/`:

1. **Missing `gguf>=0.10.0` in requirements.txt**: The model had no `requirements.txt`, so the `gguf` package was not installed. `transformers.modeling_gguf_pytorch_utils.get_gguf_hf_weights_map` raises `ImportError` if `gguf` is not installed.

2. **Session contamination from 26 narrow-sig `_patched_load_gguf_checkpoint` patches**: Twenty-six other GGUF loaders in `tt_forge_models` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at **module import time** with a narrow signature `(gguf_path, return_tensors=False)`. Pytest imports all model modules during test collection, installing the narrow-sig patch. Transformers 5.2.0 calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)` in `from_pretrained`; the narrow-sig patch raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

3. **Missing `chat_template` guard**: The GGUF tokenizer for this model has no embedded `chat_template`. Calling `self.tokenizer.apply_chat_template(...)` unconditionally raises `ValueError: tokenizer.chat_template is not set`.

## Fix
All fixes are in `tt_forge_models` on branch `remediation/huihui_qwen_3_8b_abliterated_v2_i1_gguf-causal_lm-pytorch-8B_Abliterated_v2_GGUF-single_device-inference` (commits 17312c693c, 74c97858b0, b3a63b069a).

1. **`huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch/requirements.txt`** — created with `gguf>=0.10.0`.

2. **26 GGUF loaders** (`tvall43_qwen3_5_*`, `unified_reward_flex_qwen35_27b`, `gpt_oss_swallow_120b_rl_v0_1`, `mradermacher_vilm_0_8b_sft`, `mradermacher_qwen3_5_*`, `gpt_oss_swallow_20b_rl_v0_1`, `qwen_3_5_imatrix_gguf`, `dmind_3_mini_i1_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1`, `mradermacher_crow_4b_*`, `mradermacher_bartleby_qwen3_5_4b`, `mradermacher_luna_qwen3_5_27b_v5_i1`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1`, `mradermacher_qwen3_5_27b*`, `mradermacher_qwen_3_5_27b_tainted`, `mradermacher_qwen3_5_4b_*`, `mradermacher_qwen3_5_9b_abliterated_i1`): Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(*args, **kwargs)` and `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)` in each `loader.py`.

3. **`huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch/loader.py`** — guarded `apply_chat_template` with `if self.tokenizer.chat_template is not None`, falling back to `sample_text` directly when no template is embedded.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    392.79s (0:06:32)
- Tier A attempts: N/A

## Files changed
- `huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch/requirements.txt` (created)
- `huihui_qwen_3_8b_abliterated_v2_i1_gguf/causal_lm/pytorch/loader.py`
- 26 GGUF loader `loader.py` files (narrow-sig patch widened)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 042c794f7316904a19ccaf5f4d0c28f7de77cbf0 |
| tt-forge-models | b3a63b069a610ce8eccf786ba9b24fb7503c4a0e |
