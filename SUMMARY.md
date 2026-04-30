# Remediation Summary: kimi_k2_5-pytorch-Kimi-K2.5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2_5/pytorch-Kimi-K2.5-single_device-inference]

## Result
XFAIL — Hardware capacity: Kimi K2.5 text backbone is DeepSeek-V3 class (61 layers, hidden_size=7168, 384 routed experts), ~1T total params at BF16 far exceeds all single-device DRAM.

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-kimi-k2-5-1t-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}): got AttributeError("'ndarray' object has no attribute 'add'")

In modeling_deepseek.py line 583:
  end_idx = start_idx + num_tokens
where num_tokens is a numpy scalar from .cpu().numpy().

## Root cause
The proximate failure is the moe_infer numpy dispatch bug (same class as DeepSeek V3 / Kimi K2): `num_tokens` comes from `bincount().cpu().numpy()`, yielding a numpy scalar. TT's `TorchFunctionOverride` wraps `add`, but numpy scalars have no `.add` attribute, crashing dynamo FakeTensor tracing.

However, this failure only occurs because the loader uses **forbidden trimming**: it replaces the real 61-layer/hidden_size=7168/384-expert model config with a random-weight 2-layer/1024-hidden stub. Without the trimming, the test would fail much earlier due to hardware capacity.

The real model `moonshotai/Kimi-K2.5` has a DeepSeek-V3 text backbone (61 layers, hidden_size=7168, 384 routed experts, moe_intermediate_size=2048). At BF16 the text backbone alone is ~600+ GB; the full VLM is described as ~1T parameters. This exceeds n150 (12 GB), n300 (24 GB), and p150b (32 GB) single-device DRAM by 2–3 orders of magnitude. Hardware-class XFAIL is the correct disposition.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry for `kimi_k2_5/pytorch-Kimi-K2.5-single_device-inference` in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

The loader's forbidden trimming is a secondary issue. It should be replaced with a proper `from_pretrained` call and a moe_infer static patch (same pattern as deepseek_v3/kimi_k2 loaders), but that will not change the disposition: the full model still cannot run on any supported single device.

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    59.72s
- Tier A attempts: N/A

## Files changed
- tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fa07d1ac95c894b36bd8df94d87d8cba3b0ef897 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
