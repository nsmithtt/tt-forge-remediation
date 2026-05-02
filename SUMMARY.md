# Remediation Summary: kimi_k2_5_mlx-pytorch-Kimi-K2.5-MLX-3.6bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2_5_mlx/pytorch-Kimi-K2.5-MLX-3.6bit-single_device-inference]

## Result
XFAIL — 1.026T-parameter model (438 GB on disk) exceeds all single-device DRAM limits

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-kimi-k2-5-1t-param-model

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
third_party/tt_forge_models/kimi_k2_5_mlx/pytorch/loader.py:142: in load_model
    text_config = config.text_config
venv/lib/python3.12/site-packages/transformers/configuration_utils.py:164: in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'DeepseekV3Config' object has no attribute 'text_config'. Did you mean: 'get_text_config'?
```

## Root cause
Two issues were identified:

**Loader bug**: `inferencerlabs/Kimi-K2.5-MLX-3.6bit` is a text-only model whose `config.json` sets `"AutoConfig": "configuration_deepseek.DeepseekV3Config"`. `AutoConfig.from_pretrained()` therefore returns a `DeepseekV3Config` directly — there is no VLM wrapper with a `.text_config` attribute. The loader was written assuming the same wrapper structure as the base `moonshotai/Kimi-K2.5` VLM config. The `config.text_config` access raised `AttributeError`.

**Forbidden trimming**: The original loader also overrode `num_hidden_layers=2`, `hidden_size=1024`, `num_attention_heads=16`, etc., and created a randomly initialized model via `model_class(text_config)` rather than loading from pretrained weights. This is a forbidden workaround per the remediation rules.

**Hardware capacity**: The actual model has 1,026,408,232,448 (≈ 1.026T) parameters stored in 48 safetensors shards totalling 470,011,424,768 bytes (≈ 438 GB). The p150b single-device DRAM limit is 32 GB. Loading the full model exceeds hardware capacity by ~14×.

## Fix
**In `tt_forge_models` (`kimi_k2_5_mlx/pytorch/loader.py`):**
- Replaced `config.text_config` access with `AutoModelForCausalLM.from_pretrained` which handles the auto_map from `config.json` directly.
- Removed all forbidden config trimming (`num_hidden_layers`, `hidden_size`, `num_attention_heads`, etc.) and the random-weight initialization `model_class(text_config)`.
- Removed now-unused `AutoConfig` import and `num_layers` parameter.

**In `tt-xla` (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):**
- Added `KNOWN_FAILURE_XFAIL` entries for both `Kimi-K2.5-MLX-3.6bit` and `Kimi-K2.5-MLX-4.2bit` variants citing the hardware capacity reason.

## Verification
- pytest exit: XFAIL (xfailed, hardware-capacity xfail runs in 19s)
- Hardware: wormhole (n150 via local device)
- Duration: 19.16s (xfail, no model loading attempted at full scale)
- Tier A attempts: N/A

## Files changed
- `kimi_k2_5_mlx/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 60ed68a2549ccf8bfb3385b063288b659020a748 |
| tt-forge-models | a19e2403b1a1e7bb8dc03df89a811ccde83610a4 |
