# Remediation Summary: fxmarty_qwen_1_5_moe_a2_7b_mxfp4-causal_lm-pytorch-A2.7B_MXFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fxmarty_qwen_1_5_moe_a2_7b_mxfp4/causal_lm/pytorch-A2.7B_MXFP4-single_device-inference]

## Result
XFAIL — AMD Quark MXFP4 loader incompatible with PyTorch 2.7.0 / transformers 5.2.0; even if fixed, 14.3B-param model dequantized to BF16 is 28.6 GB, exceeding p150b DRAM (24 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
mxfp4-amd-quark-unavailable-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: Quark is not installed. Please refer to https://quark.docs.amd.com/latest/install.html.
```

Raised at `transformers/utils/quantization_config.py:1843` in `QuarkConfig.__init__` when
`AutoModelForCausalLM.from_pretrained` encounters the model's MXFP4 `quantization_config`
(`quant_method: quark`, `dtype: fp4`, `scale_format: e8m0`, `group_size: 32`).

## Root cause
Two issues compound to make this model unrunnable on any single TT device:

**Issue 1 — AMD Quark loader incompatibility (Tier B loader bug).**
`fxmarty/qwen_1.5-moe-a2.7b-mxfp4` was quantized with AMD Quark MXFP4 (4-bit microscaling
floating point). Transformers 5.2.0 dispatches to `QuarkHfQuantizer` which requires the
`amd-quark` package (PyPI name `amd-quark`, import as `quark`). The `quark` package on PyPI
(version 1.0.0) is an unrelated OpenStack project — it does NOT provide `quark.torch`. The
correct AMD Quark package (`amd-quark`) has two further incompatibilities identified in the
related `fxmarty_qwen1_5_moe_a2_7b_chat_w_fp4_a_fp6_e2m3` report (fingerprint
`quark-scale-load-broken-transformers5`):
- `amd-quark==0.10` fails to import: PyTorch 2.7.0 removed
  `torch.onnx._internal.jit_utils`, used by `quark.torch.kernel` at module level.
- `amd-quark==0.11` installs but `QuarkHfQuantizer.get_weight_conversions()` uses wrong key
  patterns (`target_patterns=["weight_scale"]` → should be `["weight_quantizer.scale"]`),
  leaving all scale tensors uninitialized after loading.

**Issue 2 — hardware capacity ceiling (primary, definitive).**
Qwen 1.5-MoE has 60 experts per layer × 24 layers with `moe_intermediate_size=1408` and
`hidden_size=2048`. Total parameter count ≈ 14.3B. TT hardware has no native FP4 inference
support, so weights must be dequantized to BF16: 14.3B × 2 bytes = 28.6 GB. This exceeds
both n150 (12 GB) and p150b (24 GB) single-device DRAM. The model cannot run on any
supported single TT device.

## Fix
Marked `KNOWN_FAILURE_XFAIL` in
`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

The loader bug (Issue 1) requires coordinated upstream fixes in `transformers` and `amd-quark`
(Tier B, cross-cutting). Even if those are resolved, Issue 2 (hardware capacity) would still
prevent the model from running on any single TT device.

## Verification
- pytest exit: xfailed (1 xfailed, 7 warnings)
- Hardware:    not-run (hardware capacity ceiling, never reached silicon)
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
  (remediation branch: `b1a48c63eced9cc0a0c0f2340c61701b068f6834`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b1a48c63eced9cc0a0c0f2340c61701b068f6834 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
