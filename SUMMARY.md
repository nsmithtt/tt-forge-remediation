# Remediation Summary: eva-qwen3-next-gguf-causal-lm-pytorch-v0-0-i1-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[eva_qwen3_next_gguf/causal_lm/pytorch-v0.0_i1_GGUF-single_device-inference]

## Result
XFAIL — EVA Qwen3-Next GGUF has 79.7B parameters (~159 GB BF16), exceeding all single-device DRAM; also fixed missing qwen3next GGUF arch registration in loader

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-qwen3next-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture qwen3next is not supported yet.
```

The GGUF file declares `general.architecture = "qwen3next"` but transformers
5.2.0 had no entry for this architecture in `GGUF_CONFIG_MAPPING` (defined in
`transformers/integrations/ggml.py`). The validation at
`modeling_gguf_pytorch_utils.py:478` raised `ValueError` before any model
weights were loaded.

## Root cause
Two issues:

**Loader bug**: `qwen3next` is missing from `GGUF_TO_TRANSFORMERS_MAPPING["config"]`
(i.e. `GGUF_CONFIG_MAPPING`). The gguf-py library already supports
`MODEL_ARCH.QWEN3NEXT` for weight-name mapping, and transformers 5.2.0 already
includes `Qwen3NextForCausalLM` (model_type `qwen3_next`), but no GGUF
config mapping bridging the two existed.

**Hardware capacity**: The model has 79,674,391,296 parameters (~79.7B). At
BF16 that is 159.3 GB. No single TT device has sufficient DRAM:
- n150: 12 GB
- n300: 24 GB
- p150b: 32 GB

Even with Q4_K_M quantization the GGUF file is 46 GB on disk; after
dequantization to BF16 during inference it requires ~159 GB.

## Fix
Two changes, both in the loader layer:

**Loader patch** (`eva_qwen3_next_gguf/causal_lm/pytorch/loader.py` in
tt-forge-models):
- Added `_patch_qwen3next_support()` that appends `"qwen3next"` to
  `GGUF_SUPPORTED_ARCHITECTURES` and registers a config field mapping under
  key `"qwen3next"` in `GGUF_TO_TRANSFORMERS_MAPPING["config"]`.
- The mapping covers: `block_count`, `context_length`, `embedding_length`,
  `feed_forward_length`, `attention.head_count`, `attention.head_count_kv`,
  `attention.key_length`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`,
  `vocab_size`, `expert_count`, `expert_used_count`,
  `expert_feed_forward_length`, `expert_shared_feed_forward_length`,
  `ssm.conv_kernel`, `full_attention_interval`.
- `_patched_load_gguf_checkpoint` fixes `model_type` from `"qwen3next"` to
  `"qwen3_next"` so `AutoModelForCausalLM` selects `Qwen3NextForCausalLM`.
- `_patched_get_gguf_hf_weights_map` remaps `"qwen3_next"` → `"qwen3next"`
  for gguf-py tensor-name lookup (which already supports `MODEL_ARCH.QWEN3NEXT`).

**Test config** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in tt-xla):
- Added `KNOWN_FAILURE_XFAIL` entry for the test with hardware-capacity reason.

## Verification
- pytest exit: XFAIL
- Hardware:    not-run (test calls pytest.xfail before reaching silicon)
- Duration:    88.51s (collection + xfail path)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/eva_qwen3_next_gguf/causal_lm/pytorch/loader.py` — qwen3next GGUF arch registration
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e6693698df0e3c75c608e0d2f862d3b89e4c43a7 |
| tt-forge-models | 81709ceaa1720eef02fd36edaf93afbb4d42c5a4 |
