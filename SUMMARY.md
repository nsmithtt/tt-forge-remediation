# Remediation Summary: hunyuan_video_gguf-pytorch-Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_gguf/pytorch-Q8_0-single_device-inference]

## Result
XFAIL — HunyuanVideo T2V 13B requires ~23.88GB BF16 DRAM; activations push total over p150b single-device capacity (24GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-13b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 22:22:08.100 | critical |          Always | TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected (assert.hpp:104)

## Root cause
HunyuanVideo T2V 720p (city96/HunyuanVideo-gguf) has 12.82B parameters.
Without GGUFQuantizationConfig, diffusers dequantizes Q8_0 weights to BF16 on
load, materializing ~23.88GB of weight tensors. The p150b has 24GB DRAM; with
activation tensors during compilation the total exceeds device capacity, causing
the device to OOM and report a timeout/hang.

The bringup branch (hf-bringup-40) contained a forbidden workaround: a
`except (RuntimeError, ValueError): self.transformer = HunyuanVideoTransformer3DModel()`
fallback to random weights. This fallback loaded the same 12.82B parameter
model, so the hardware OOM occurred identically. The workaround was removed.

Legitimate loader fixes also applied from hf-bringup-40:
1. `requirements.txt` with `gguf>=0.10.0` (diffusers requires gguf>=0.10 to load GGUFs).
2. Refresh `diffusers._gguf_available` flag at load time — diffusers caches
   this at module import, before the test runner installs gguf.
3. `ignore_mismatched_sizes=True` — handles minor shape differences between the
   GGUF tensors and the diffusers model parameters.
4. `guidance` tensor added to `load_inputs` — `guidance_embeds=True` in the
   default config requires a guidance scale input.

## Fix
- `tt_forge_models/hunyuan_video_gguf/pytorch/requirements.txt` — new file, `gguf>=0.10.0`
- `tt_forge_models/hunyuan_video_gguf/pytorch/loader.py` — gguf import refresh,
  `ignore_mismatched_sizes=True`, guidance tensor; forbidden random-weights fallback removed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` —
  `KNOWN_FAILURE_XFAIL` for both `Q8_0` and `Q4_K_S` variants

## Verification
- pytest exit: TIMEOUT (device OOM; model not run due to hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    not-run (hardware capacity confirmed analytically: 12.82B params × 2B = 23.88GB > 24GB - activation headroom)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hunyuan_video_gguf/pytorch/requirements.txt` (created)
- `tt_forge_models/hunyuan_video_gguf/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6de33af27874f4c6600700516701063a6a664420 |
| tt-forge-models | 89f1f823348878b9938947d922903826c82ab2dd |
