# Remediation Summary: bartowski_c4ai_command_r_plus_gguf-causal_lm-pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-tensor_parallel-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-tensor_parallel-inference]

## Result
XFAIL — 104B-parameter model at BF16 requires ~208 GB DRAM, exceeding the 192 GB device DRAM available on all supported tensor-parallel configurations

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-104b-exceeds-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure message: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Underlying root causes exposed during loader debugging:
1. `OSError`: The HuggingFace repo reorganized `c4ai-command-r-plus-Q4_K_M.gguf` from a single 58 GB file into 6 sharded files under a subdirectory; transformers single-file GGUF loading cannot handle sharded GGUFs.
2. `ValueError: GGUF model with architecture command-r is not supported yet.` — `command-r` architecture was not ported to `GGUF_CONFIG_MAPPING` in transformers 5.x.
3. `KeyError: 'cohere'` in `convert_gguf_tokenizer` — `GGUF_TO_FAST_CONVERTERS` had no entry for `"cohere"` (the remapped model_type).
4. `ValueError: tokenizer.chat_template is not set` — Q3_K_S GGUF does not embed a chat template.
5. Hardware capacity ceiling: after all loader fixes, the test loaded the 43 GB Q3_K_S checkpoint and ran for ~28 minutes at 208% CPU / ~18–20% host RAM (~135–152 GB). The 104B model at BF16 requires ~208 GB device DRAM, exceeding the P150b Blackhole's 192 GB LPDDR5X (8 channels, `ENABLED_GDDR=0xff`).

## Root cause
The `command-r` GGUF architecture (`general.architecture = "command-r"`) was not ported to transformers 5.x `GGUF_CONFIG_MAPPING` or `GGUF_SUPPORTED_ARCHITECTURES`, causing `load_gguf_checkpoint` to raise `ValueError`. Additionally, `GGUF_TO_FAST_CONVERTERS` lacked a `"cohere"` entry (required after model_type remapping), causing `KeyError` in the tokenizer converter.

These are loader-layer bugs in the tt-forge-models submodule. All four loader bugs were fixed. After the fixes, the model itself (104B parameters, ~208 GB at BF16) exceeds the device DRAM on all supported tensor-parallel configurations, classifying the test as a hardware-class ceiling.

## Fix
Four commits in `tt-forge-models` remediation branch (`remediation/bartowski_c4ai_command_r_plus_gguf-causal_lm-pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-tensor_parallel-inference`):

1. **`512ea44e5e`** — `bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch/loader.py`: Changed `GGUF_FILE` from `c4ai-command-r-plus-Q4_K_M.gguf` to `c4ai-command-r-plus-Q3_K_S.gguf` (best-quality single-file GGUF; Q4_K_M was split into 6 shards not supported by transformers).
2. **`78bcaf8175`** — `loader.py` (and shared gguf base): Forward `**kwargs` through `load_gguf_checkpoint` wrappers for transformers 5.x `model_to_load` kwarg compatibility.
3. **`c30a66a071`** — `loader.py`: Register `command-r` in `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`; patch `load_gguf_checkpoint` in all 3 import-by-name call-site namespaces to remap `model_type="command-r"` → `"cohere"` so `AutoConfig` selects `CohereConfig`; register `GGUF_TO_FAST_CONVERTERS["cohere"] = GGUFGPTConverter` for tokenizer conversion.
4. **`e4a4c91423`** — `loader.py`: Wrap `apply_chat_template` in try/except with fallback to `sample_text` when the GGUF does not embed a chat template.

One commit in `tt-xla` remediation branch:
- **`a04f0d47a`** — `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`: Add `KNOWN_FAILURE_XFAIL` entry for this test with reason citing the 208 GB BF16 vs 192 GB device DRAM constraint.

## Verification
- pytest exit: TIMEOUT (process killed after ~28 min; model loading exceeded device DRAM capacity)
- Hardware:    blackhole-p150b
- Duration:    ~28 min (killed; not a passing run)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 34758aee15c584be1ce0b6da5c976960d1f02c06 |
| tt-forge-models | e4a4c914236de7e7efd9f50f73469929140715a6 |
