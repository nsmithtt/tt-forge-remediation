# Remediation Summary: aisingapore_gemma_sea_lion_v4_27b_it-causal_lm-pytorch-Gemma_SEA_LION_v4_27B_IT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aisingapore_gemma_sea_lion_v4_27b_it/causal_lm/pytorch-Gemma_SEA_LION_v4_27B_IT-single_device-inference]

## Result
XFAIL — 27B BF16 model (~54 GB) exceeds all single-device DRAM (n150: 12 GB, p150b: 24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gemma3-27b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The swigvarlink warning is emitted by a SWIG-based extension at Python startup;
the underlying test failure captured locally is:
TypeError: Gemma3ForConditionalGeneration.__init__() got an unexpected keyword argument 'use_cache')

## Root cause
Two issues:

1. **Loader bug (fixed)**: The loader passes `use_cache=False` directly to
   `AutoModelForCausalLM.from_pretrained(...)`. For this model,
   `AutoModelForCausalLM` resolves to `Gemma3ForConditionalGeneration` (the VLM
   class) because the config's `architectures` field specifies that class.
   `Gemma3ForConditionalGeneration.__init__` does not accept `use_cache` as a
   constructor kwarg, causing a `TypeError`. The fix sets `use_cache=False` on
   `config.text_config` before calling `from_pretrained`.

2. **Hardware capacity (XFAIL)**: `aisingapore/Gemma-SEA-LION-v4-27B-IT` is a
   27B-parameter Gemma 3 model (text decoder: 62 layers, hidden_size=5376).
   In BF16: 27B × 2 bytes ≈ 54 GB. This exceeds n150 DRAM (12 GB) and p150b
   DRAM (24 GB). No single TT device can hold this model.

## Fix
- **tt-forge-models** `aisingapore_gemma_sea_lion_v4_27b_it/causal_lm/pytorch/loader.py`:
  Replaced `model_kwargs = {"use_cache": False}` with a config-based approach
  that calls `AutoConfig.from_pretrained` first, sets
  `config.text_config.use_cache = False` (or `config.use_cache` for flat
  configs), and passes `config=config` to `from_pretrained`.

- **tt-xla** `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `KNOWN_FAILURE_XFAIL` entry for this test.

## Verification
- pytest exit: FAIL (TypeError before reaching silicon; after loader fix the model would OOM on device)
- Hardware:    not-run
- Duration:    88.35s (local run, failed at loader TypeError)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/aisingapore_gemma_sea_lion_v4_27b_it/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2d0ad16522dd9cd29b9e1764468382439aa7989f |
| tt-forge-models | d515976a61bbdcd7477d66928208f2ad9eaee4c0 |
