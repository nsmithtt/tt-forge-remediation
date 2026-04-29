# Remediation Summary: gemma-text_translation-pytorch-VLLM_Translategemma_27B_IT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma/text_translation/pytorch-VLLM_Translategemma_27B_IT-single_device-inference]

## Result
XFAIL — 27B BF16 model (~54 GB) exceeds single p150b device DRAM (~32 GB); test config updated to KNOWN_FAILURE_XFAIL

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-27b-bf16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
E   RuntimeError: Value out of range (expected to be in range of [-91, 90], but got -1023)
```

Additional failures discovered during reproduction (loader bugs, then DRAM OOM):
1. `jinja2.exceptions.TemplateError: User role must provide content as a string in format: '<<<source>>><lang_code><<<target>>><lang_code><<<text>>><text_to_translate>'`
2. `TypeError: string indices must be integers, not 'str'` (Gemma3Processor.apply_chat_template visual extraction)
3. `AttributeError: 'tuple' object has no attribute 'past_key_values'` (return_dict=False propagated to internal model call)
4. `TT_FATAL: Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks, where each bank needs to store 28901376 B, but bank size is 4273390016 B (allocated: 4225215296 B, free: 48174720 B)`

## Root cause
**Three loader bugs** prevented the model from reaching device execution:

1. **Wrong content format**: `sample_messages` used list-of-dicts content but the `Infomaniak-AI/vllm-translategemma-27b-it` model has a custom Jinja2 chat template that requires plain string content in `<<<source>>>cs<<<target>>>de-DE<<<text>>>...` format.

2. **Processor visual extraction**: `Gemma3Processor.apply_chat_template` runs visual extraction before Jinja rendering (processing_utils.py:1755), iterating over `message["content"]` assuming a list. A plain string breaks this. Fix: bypass the multimodal processor and use `processor.tokenizer.apply_chat_template` + `processor.tokenizer()` directly.

3. **return_dict=False propagation**: `load_model` passes `return_dict=False` to `from_pretrained`, setting `config.return_dict=False`. Inside `Gemma3ForConditionalGeneration.forward`, `self.model(...)` at line 1083 is called without explicit `return_dict=True` and uses the config default (False), returning a tuple. The outer `forward` then does `outputs.past_key_values` on the tuple, raising `AttributeError`.

After fixing the three loader bugs, the model reaches TT device execution but hits DRAM OOM: a 27B BF16 model requires ~54 GB but the p150b device has ~32 GB DRAM (8 banks × ~4 GB each).

The original CI failure (`Value out of range`) is the `aten.slice` XLA lazy-backend OOB error — PyTorch eager silently clamps out-of-range negative indices (e.g. a sliding window of 4096 on a 91-token sequence gives start=-4005), but XLA raises instead of clamping. This fix was also applied to `torch_overrides.py` (same fix confirmed by prior session for albert-wesker and big_tiger_gemma tests).

## Fix
**tt_forge_models** (`remediation/gemma-text_translation-pytorch-VLLM_Translategemma_27B_IT-single_device-inference`):
- `gemma/text_translation/pytorch/loader.py`:
  - Added `vllm_sample_messages` with string content in `<<<source>>>cs<<<target>>>de-DE<<<text>>>...` format
  - `load_inputs`: for VLLM variant, use `processor.tokenizer.apply_chat_template` + `processor.tokenizer()` to bypass Gemma3Processor visual extraction
  - `load_model`: omit `return_dict=False` for VLLM variant to prevent internal call OOB on tuple result

**tt-xla** (`remediation/gemma-text_translation-pytorch-VLLM_Translategemma_27B_IT-single_device-inference`):
- `python_package/tt_torch/torch_overrides.py`: Added `aten.slice.Tensor` interception to pre-clamp negative start/end indices that are out of range (`< -size`), matching PyTorch eager behaviour
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test with DRAM capacity reason

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — marked XFAIL (hardware-capacity)

## Verification
- pytest exit: FAIL (OOM on device — hardware capacity ceiling)
- Hardware:    blackhole-p150b
- Duration:    470.59s (0:07:50) until DRAM OOM
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/gemma/text_translation/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fc2f32038cfebd3e176966bccfa37f8d930d7589 |
| tt-forge-models | e3a8d267074763697d696291835f1422ccd9fb47 |
