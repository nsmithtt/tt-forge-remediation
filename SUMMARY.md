# Remediation Summary: indextts_1_5-pytorch-IndexTTS-1_5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[indextts_1_5/pytorch-IndexTTS-1.5-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
indextts-non-module-device-placement

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_module L__self___gpt_gpt_inference_model_transformer_h_0_ln_1(*(FakeTensor(..., device='xla:0', size=(1, 32, 1280)),), **{}): got RuntimeError('Unhandled FakeTensor Device Propagation for aten.addcmul.default, found two different devices cpu, xla:0')

## Root cause
`IndexTTS` (from the `indextts` package) is a plain Python class — it does **not** inherit from `nn.Module`. The original `IndexTTS15GPTWrapper` stored the entire `IndexTTS` object as `self.gpt`. When PyTorch's `nn.Module.__setattr__` encounters a non-`Module` object, it stores it as a plain dict entry, bypassing the submodule registry. As a result, calling `.to(xla_device)` on the wrapper silently left all parameters of `tts.gpt.inference_model.transformer` (the GPT2Model) on CPU. Dynamo then encountered a device mismatch when it tried to trace through `ln_1` (LayerNorm) with XLA inputs against CPU weights, surfacing as an `aten.addcmul.default` device propagation failure.

## Fix
In `tt-forge-models` `indextts_1_5/pytorch/loader.py`:

1. Changed `IndexTTS15GPTWrapper.__init__` to accept an `inference_model` (a `GPT2InferenceModel`, which IS an `nn.Module`) instead of the full `IndexTTS` object. Storing it as `self.inference_model` properly registers it as a submodule so `.to(device)` moves all parameters.

2. Updated `forward()` to call `self.inference_model.transformer` and `self.inference_model.lm_head` directly (removing the intermediate `tts.gpt.inference_model` lookup).

3. Updated `load_model()` to extract `inference_model = tts.gpt.inference_model` before constructing the wrapper.

Remediation branch: `remediation/indextts_1_5-pytorch-IndexTTS-1_5-single_device-inference` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    160.63s (0:02:40)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/indextts_1_5/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 29ceba10be9d37203d9f7247e41fb0bf83f8e433 |
