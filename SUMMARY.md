# Remediation Summary: granite_4_0_h_1b_gguf-causal_lm-pytorch-granite_4_0_h_1b_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_h_1b_gguf/causal_lm/pytorch-granite_4_0_h_1b_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-granitehybrid-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-24 00:18:59.922 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
Two loader bugs in the `granite_4_0_h_1b_gguf` model:

**Bug 1 — arch not registered:** `granitehybrid` was absent from
`GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`,
`CONFIG_MAPPING`, and `GGUF_TO_FAST_CONVERTERS` in transformers 5.x.
`AutoModelForCausalLM.from_pretrained` with `gguf_file=` either failed
silently or loaded with a mismatched config, corrupting weights; the device
subsequently timed out during the forward pass. Also: `gguf>=0.10.0` was
missing from requirements.txt, and `load_gguf_checkpoint` had the narrow-sig
incompatibility with transformers 5.x's new `model_to_load` kwarg.

**Bug 2 — tensor shape mismatches in GGUFTensorProcessor:** Without a custom
`TensorProcessor` for `granitehybrid`, four shape transformations were missing:
  - `ssm_conv1d.weight`: gguf-py gives `[C, K]`, HF expects `[C, 1, K]`
  - `ssm_a` / `ssm_d`: gguf-py gives `[n, 1]`, HF expects `[n]`
  - `ffn_gate` + `ffn_up` → `shared_mlp.input_linear`: GGUF stores gate and up
    separately; HF uses a single concatenated `[2*I, H]` tensor
  - `ffn_down` → `shared_mlp.output_linear`: gguf-py maps `output_linear` to
    `ffn_down_shexp` but the GGUF file uses `ffn_down` (no `_shexp` suffix);
    the weight was silently skipped, leaving `output_linear.weight` uninitialized
    for all 40 layers

Together, wrong weights in the Mamba2 SSM and shared-MLP layers caused the
forward pass to hang on the TT device.

## Fix
All fixes in `tt-forge-models` on branch
`remediation/granite_4_0_h_1b_gguf-causal_lm-pytorch-granite_4_0_h_1b_Q4_K_M_GGUF-single_device-inference`:

- `granite_4_0_h_1b_gguf/causal_lm/pytorch/loader.py` —
  - `_register_granitehybrid_gguf_support()`: registers `granitehybrid` in all
    four transformers GGUF maps; also patches `get_gguf_hf_weights_map` to remap
    `granitemoehybrid→granitehybrid` for gguf-py lookup
  - `_GraniteHybrid1BTensorProcessor.process()`: applies the four shape fixes
    described above via regex-matched tensor name handlers
  - `_GraniteHybrid1BTensorProcessor.perform_fallback_tensor_mapping()`: adds
    GGUF→HF name mappings for `shared_mlp.input_linear`, `shared_mlp.output_linear`,
    and `mamba.dt_bias`
  - `load_model()`: loads base config from `ibm-granite/granite-4.0-h-1b`
    (non-GGUF) to obtain `layer_types`; applies narrow-sig `model_to_load` patch
  - `load_config()`: likewise uses the base model for config

- `granite_4_0_h_1b_gguf/causal_lm/pytorch/requirements.txt` — adds `gguf>=0.10.0`

Commits: `93c2e52c0f`, `899ca4b3c3`, `12903bba3f`, `57ea730a49`, `4ffb99d19c`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    420.65s (0:07:00)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/granite_4_0_h_1b_gguf/causal_lm/pytorch/loader.py`
- `tt-forge-models/granite_4_0_h_1b_gguf/causal_lm/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1278fdbfe8038f1ecb068cafb1f5d570ba8e2c17 |
| tt-forge-models | 4ffb99d19cb420705cb2df65737d604b2101e2f3 |
