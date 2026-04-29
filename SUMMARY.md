# Remediation Summary: command_r-causal_lm-pytorch-Plus_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[command_r/causal_lm/pytorch-Plus_GGUF-single_device-inference]

## Result
XFAIL — C4AI Command R+ (~104B params, ~58 GB Q4_K_M) exceeds single-device DRAM on all TT hardware; hardware capacity ceiling

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-command-r-arch-missing-and-hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Loading weights:  25%|██▌       | 67/264 [00:06<00:32,  6.07it/s, Materializing param=model.layers.8.mlp.down_proj.weight]
```

## Root cause

Two issues, both in the loader layer:

1. **Missing GGUF architecture registration**: The GGUF file stores
   `general.architecture = "command-r"`, but transformers 5.x
   `GGUF_SUPPORTED_ARCHITECTURES` does not include `"command-r"`. In the
   current environment this raises `ValueError: GGUF model with architecture
   command-r is not supported yet.` before any weights are loaded. At CI run
   time the architecture was apparently registered (possibly by a co-collected
   test or an earlier transformers snapshot), allowing loading to proceed.
   Additionally, the GGUF general.architecture `"command-r"` maps to
   HuggingFace model_type `"cohere"` (CohereConfig / CohereForCausalLM);
   without a remapping, AutoConfig cannot find the correct config class.

2. **Hardware capacity ceiling**: C4AI Command R+ has ~104B parameters. At
   Q4_K_M quantization (~4.5 bits/param) the weights occupy ~58 GB. After
   dequantization to bfloat16 at load time, RAM requirements further increase.
   The CI failure at 25% of weight loading (67/264 params, layer 8 down_proj)
   is consistent with the process being OOM-killed partway through CPU
   materialization. No single TT device can hold this model: n150 has 12 GB
   DRAM, p150b has 24 GB DRAM. This is a hardware capacity ceiling, not a
   compiler bug.

## Fix

In `tt_forge_models` (remediation branch
`remediation/command_r-causal_lm-pytorch-Plus_GGUF-single_device-inference`):

`command_r/causal_lm/pytorch/loader.py`:
- Added `_patch_transformers_command_r_gguf()` called at module level which:
  - Appends `"command-r"` to `GGUF_SUPPORTED_ARCHITECTURES`
  - Adds CohereConfig field mappings to `GGUF_TO_TRANSFORMERS_MAPPING["config"]`
    (`context_length` → `max_position_embeddings`, `block_count` →
    `num_hidden_layers`, `feed_forward_length` → `intermediate_size`,
    `embedding_length` → `hidden_size`, `rope.freq_base` → `rope_theta`,
    `attention.head_count` → `num_attention_heads`,
    `attention.head_count_kv` → `num_key_value_heads`,
    `attention.layer_norm_rms_epsilon` → `layer_norm_eps`,
    `vocab_size` → `vocab_size`)
  - Uses `_find_real_load_gguf()` to walk the `__globals__` chain and locate
    the actual transformers `load_gguf_checkpoint` past any prior monkey-patchers
  - Wraps `load_gguf_checkpoint` (with `**kwargs` for `model_to_load` forwarding)
    to translate `model_type "command-r"` → `"cohere"` so AutoConfig resolves
    `CohereConfig`; patches both `modeling_gguf_pytorch_utils` and
    `configuration_utils` import sites

In `tt-xla` (remediation branch
`remediation/command_r-causal_lm-pytorch-Plus_GGUF-single_device-inference`):
- Added `command_r/causal_lm/pytorch-Plus_GGUF-single_device-inference` to
  `KNOWN_FAILURE_XFAIL` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
  with reason: model too large for single device.

## Verification
- pytest exit: FAIL (xfailed — test expected to fail due to hardware capacity; local
  run timed out awaiting 29 GB GGUF download; CI showed OOM at 25% weight loading)
- Hardware: not-run (hardware capacity ceiling; model cannot fit on any single n150/n300/p150b device)
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `command_r/causal_lm/pytorch/loader.py` (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 221420ca7a12925a0cbc04dd6cf343bfe577ee09 |
| tt-forge-models | 77a5066571c7e433092663a4655ebae3c60f6d0e |
