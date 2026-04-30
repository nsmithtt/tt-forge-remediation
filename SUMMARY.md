# Remediation Summary: glm_4_7_flash_uncensored_hauhaucs_balanced_gguf-causal_lm-pytorch-4.7_Flash_Uncensored_HauhauCS_Balanced_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_7_flash_uncensored_hauhaucs_balanced_gguf/causal_lm/pytorch-4.7_Flash_Uncensored_HauhauCS_Balanced_GGUF-single_device-inference]

## Result
FAIL — Tier B segfault in tt-xla CPU-fallback partitioner during graph compilation of DeepSeek-V2/GLM-4.7 MoE model

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-cpu-fallback-partition-segfault-moe

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (CI, base branch `0f7b734348`):
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

In full pytest sessions (other loaders imported first, before loader fix):
```
KeyError: 'deepseek_v2'
```
at `transformers/integrations/ggml.py:787: converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)`

After loader fixes, confirmed crash:
```
Fatal Python error: Segmentation fault
  File "torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File "tt_torch/torch_overrides.py", line 34 in __torch_function__
```

## Root cause

**Loader bugs (fixed):**

1. **Missing `requirements.txt`**: The loader had no `requirements.txt` declaring `gguf>=0.10.0`. When `gguf` is not installed, `is_gguf_available()` returns False and `load_gguf_checkpoint` raises `ImportError("Please install torch and gguf>=0.10.0...")`.

2. **Missing deepseek2 GGUF architecture support**: The GGUF file's `general.architecture` is `deepseek2`, which is not in transformers' `GGUF_SUPPORTED_ARCHITECTURES`. Without the patch, `load_gguf_checkpoint` raises `ValueError: GGUF model with architecture deepseek2 is not supported yet`.

3. **`KeyError: 'deepseek_v2'` in multi-loader sessions**: The sibling loader `glm_4_7_flash_gguf` patches `gguf_utils.load_gguf_checkpoint` to remap `model_type` from `"deepseek2"` to `"deepseek_v2"` for model loading compatibility. The transformers tokenizer reads `architecture = gguf_param["config"]["model_type"]` (line 257 of `tokenization_utils_tokenizers.py`) and passes it to `convert_gguf_tokenizer`. Since `"deepseek_v2"` is not in `GGUF_TO_FAST_CONVERTERS`, this raises `KeyError: 'deepseek_v2'`. The `hauhaucs_balanced` loader must register both keys and use a context manager to install correct loading functions.

4. **Additional loader fixes**: Corrected `qk_nope_head_dim` and combined split MLA kv projections (`attn_k_b` + `attn_v_b` → `attn_kv_b`); fixed `load_shard_spec` for DeepseekV2 MLA attention; corrected `num_key_value_heads` for MLA; replaced complex RoPE arithmetic with real cos/sin.

**Compiler bug (Tier B, unfixed):**

After all loader fixes, the test reaches the compilation stage in `tt_torch/backend/backend.py:_call_experimental_compile`. During `partition_fx_graph_for_cpu_fallback` in `torch_xla/_dynamo/dynamo_bridge.py:762`, an FX interpreter runs ops on tensors to determine TT vs CPU partitioning. A specific op in the GLM-4.7 (DeepSeek-V2) MoE model causes a C-level segmentation fault inside `TorchFunctionOverride.__torch_function__` (tt-xla `torch_overrides.py:34`). This is the same class of crash as the `qwen35moe` Tier B failure (fingerprint `pjrt-cpu-fallback-partition-segfault-moe`).

## Fix
**Loader fixes applied** (in `tt_forge_models` remediation branch `remediation/glm_4_7_flash_uncensored_hauhaucs_balanced_gguf-causal_lm-pytorch-4.7_Flash_Uncensored_HauhauCS_Balanced_GGUF-single_device-inference`):
- `glm_4_7_flash_uncensored_hauhaucs_balanced_gguf/causal_lm/pytorch/loader.py`: Added `_register_deepseek2_gguf_support()` (registers both `"deepseek2"` and `"deepseek_v2"` in `GGUF_TO_FAST_CONVERTERS`, adds config mapping, adds `GGUF_SUPPORTED_ARCHITECTURES` entry); added `_Deepseek2GlmTensorProcessor` to combine split MLA kv tensors; added `_deepseek2_gguf_load_ctx()` context manager with correct `load_gguf_checkpoint` chaining and `get_gguf_hf_weights_map` remapping; fixed MLA attention config params; replaced complex RoPE with real arithmetic.
- `glm_4_7_flash_uncensored_hauhaucs_balanced_gguf/causal_lm/pytorch/requirements.txt`: Added `gguf>=0.10.0`.

**Compiler fix proposed** (unfixed, Tier B):
The `partition_fx_graph_for_cpu_fallback` function in `torch_xla/_dynamo/dynamo_bridge.py` crashes on one or more ops from the DeepSeek-V2/GLM-4.7 MoE architecture during the CPU/TT partitioning phase. The fix requires identifying the specific op(s) that cause the segfault via C debugger or per-node try/except in the FX interpreter, then either: (a) guarding them in the partitioner's FX interpreter, (b) adding safe fallback handling in `TorchFunctionOverride`, or (c) patching the aten op registration that causes the crash.

## Tier B justification
`internal-error-unknown-mechanism`: The segfault occurs at a C-level in the PyTorch aten dispatch system during CPU-fallback partitioning. The precise op that crashes is not yet identified (requires attaching a C debugger or adding per-node try/except instrumentation to the FX interpreter). Without knowing the failing op, the fix scope cannot be estimated and diagnosis must precede any fix attempt.

## Verification
- pytest exit: FAIL (process segfault — Fatal Python error: Segmentation fault)
- Hardware:    n150
- Duration:    ~31 minutes (model loading + compilation before crash)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/glm_4_7_flash_uncensored_hauhaucs_balanced_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/glm_4_7_flash_uncensored_hauhaucs_balanced_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 70dd926372bb0aa8bcc4149bbb95c65838f6d723 |
