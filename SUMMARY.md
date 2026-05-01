# Remediation Summary: llama_3_3_nemotron_super_49b_v1_gguf-causal_lm-pytorch-Llama_3_3_Nemotron_Super_49B_v1_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_3_nemotron_super_49b_v1_gguf/causal_lm/pytorch-Llama_3_3_Nemotron_Super_49B_v1_Q4_K_M-single_device-inference]

## Result
XFAIL — Hardware capacity ceiling: NVIDIA Llama 3.3 Nemotron Super 49B Q4_K_M dequantized to BF16 requires ~98 GB DRAM, exceeding the p150b single-device limit of 96 GB.

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-deci-arch-not-in-config-mapping

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The GGUF file declares `general.architecture = "deci"` (NVIDIA/DeciLM original name). transformers 5.2.0 has `"deci"` in `GGUF_CONFIG_MAPPING` and `GGUF_TO_FAST_CONVERTERS`, but NOT in `CONFIG_MAPPING`. When any loader imported alphabetically before this one patches `_gguf_utils.load_gguf_checkpoint`, it captures the previous binding without passing through `model_to_load`, breaking the full-session loading chain.

Additionally, the model uses a NAS-optimized per-layer variable architecture:
- `num_attention_heads` is a list of 80 values (some zero — those layers have no self-attention)
- `num_key_value_heads` is a list of 80 values
- `intermediate_size` is a list of 80 values (ranging 2816–28672)

Standard transformers `LlamaConfig` expects scalar values for these fields.

At 49B parameters, even the Q4_K_M GGUF (~29 GB on disk) materializes to ~98 GB when dequantized to BF16 for device loading, exceeding the 96 GB p150b single-device DRAM capacity.

Original reported failure:
```
Loading weights:  49%|████▉     | 277/568 [00:11<00:27, 10.54it/s, Materializing param=model.layers.32.mlp.down_proj.weight]
```

Reproduced failure (without loader fix):
```
ValueError: The checkpoint you are trying to load has model type 'deci' but Transformers does not recognize this architecture.
```

## Root cause
Two root causes:

1. **Loader bug**: `"deci"` is not registered in transformers 5.x `CONFIG_MAPPING`. Any loader imported alphabetically earlier (e.g., `deepseek_v3`, `gemma3n`) patches `_gguf_utils.load_gguf_checkpoint` at module level. Later loaders that don't BFS-traverse the patch chain call these intermediate wrappers which lack the `model_to_load` kwarg added in transformers 5.x, causing `TypeError`. The Nemotron loader needed to: (a) traverse the full patch chain via `__closure__` and `__globals__` to find the true original `load_gguf_checkpoint`, (b) remap `model_type "deci" → "llama"` in the parsed config, and (c) collapse per-layer list fields to `max(non-zero)` scalars. Additionally `ignore_mismatched_sizes=True` is needed since layer FFN sizes vary below the max scalar.

2. **Hardware-class ceiling**: After the loader fix, the 49B model dequantizes to ~98 GB BF16, exceeding the 96 GB p150b single-device DRAM. This is not a compiler bug.

## Fix
**Loader fix** (in `tt_forge_models`, commit `01bd2da065`):

- `tt-xla/third_party/tt_forge_models/llama_3_3_nemotron_super_49b_v1_gguf/causal_lm/pytorch/loader.py`
  - Added `_unwrap_to_true_load_gguf_checkpoint()`: BFS traversal via `__closure__` cells and `__globals__` looking for keys containing `"orig"/"true"/"real"/"load_gguf"` to find the transformers-native function
  - Added `_patched_load_gguf_checkpoint()`: remaps `config["model_type"] = "deci"` → `"llama"`, and collapses `num_attention_heads`, `num_key_value_heads`, `intermediate_size` lists to `max(non_zero)` scalar
  - Patches all four binding sites: `_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`
  - Re-applies patches in `load_model()` in case a later import overwrites them
  - Adds `ignore_mismatched_sizes=True` to `from_pretrained` kwargs

**XFAIL config** (in `tt-xla`, commit `bb9816759`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added entry:
  ```yaml
  llama_3_3_nemotron_super_49b_v1_gguf/causal_lm/pytorch-Llama_3_3_Nemotron_Super_49B_v1_Q4_K_M-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "Hardware capacity ceiling: 49B model dequantized to BF16 requires ~98 GB which exceeds p150b single-device DRAM (96 GB)."
  ```

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    not-run (XFAIL — model exceeds device DRAM; test reports xfailed before reaching silicon)
- Duration:    1721.33s (0:28:41) — includes full GGUF download and BF16 dequantization to confirm OOM
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llama_3_3_nemotron_super_49b_v1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f44606736279677d42daed9d5363571fe2eedea2 |
| tt-forge-models | 01bd2da0655775492f857d22c62ac8eb79c3e908 |
