# Remediation Summary: mirothinker_v1_5_30b_gguf-causal_lm-pytorch-v1.5_30B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mirothinker_v1_5_30b_gguf/causal_lm/pytorch-v1.5_30B_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B based model (~64 GB BF16) exceeds single p150b DRAM capacity; KNOWN_FAILURE_XFAIL added to test config

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-qwen3-30b-a3b-single-device-dram-oom

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
Fatal Python error: Segmentation fault
```

After fixing the nonzero/for-loop segfault in Qwen3MoeExperts and the
model_to_load kwarg TypeError in GGUF loaders, the test ran to completion
and failed with:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

C++ stack bottom:
```
tt::tt_metal::distributed::MeshBuffer::create(...)
tt::tt_metal::tensor_impl::allocate_device_buffer(...)
tt::tt_metal::tensor_impl::to_device(...)
tt::runtime::ttnn::operations::layout::run(tt::target::ttnn::ToDeviceOp const*...)
tt::runtime::ttnn::ProgramExecutor::execute()
tt::runtime::ttnn::operations::cache::run(tt::target::ttnn::LoadCachedOp const*...)
tt::runtime::ttnn::submit(...)
```

## Root cause
Two bugs were found and fixed during remediation:

**Bug 1 (loader, fixed):** 26 GGUF loaders in tt_forge_models patch
`load_gguf_checkpoint` at import time with a wrapper missing the `model_to_load`
kwarg added in transformers 5.x. During pytest collection all loaders are
imported, leaving a broken global patch that triggers TypeError when the
target GGUF model eventually calls `from_pretrained`. Fixed in
`tt_forge_models` by adding `model_to_load=None` to 26 wrapper signatures.

**Bug 2 (tt-xla, fixed):** `Qwen3MoeExperts.forward` uses `nonzero()` which
segfaults during `partition_fx_graph_for_cpu_fallback` (the FX interpreter
replays ops on CPU to determine the device/CPU partition boundary). Fixed in
`tt-xla/python_package/tt_torch/torch_overrides.py` by adding
`_qwen3moe_experts_forward` with a CPU per-expert loop path and a device
dense-bmm path — same pattern as the existing `GptOssExperts` fix.

**Terminal failure (hardware-class):** After both fixes the test ran to
completion but hit INTERNAL Error code 13 from `allocate_device_buffer`
during inference. The C++ stack shows `ToDeviceOp` → `MeshBuffer::create`
failing. MiroThinker v1.5 30B is based on Qwen3-30B-A3B; in BF16 the model
is ~64 GB. The existing test config already marks the base model
(`qwen_3/causal_lm/pytorch-30B_A3b-single_device-inference`) as
`EXCLUDE_MODEL` with note "Too large for single chip, run as tensor_parallel
instead." This confirms the model exceeds single p150b (96 GB DRAM) capacity
when accounting for compiled binary size and runtime buffers.

## Fix
1. **tt_forge_models** (`remediation/mirothinker_v1_5_30b_gguf-causal_lm-pytorch-v1.5_30B_GGUF-single_device-inference`, commit `8c3a9d91f6`):
   - `third_party/tt_forge_models/*/loader.py` — 26 files: added `model_to_load=None` parameter to `_patched_load_gguf_checkpoint` and passed it to `_orig_load_gguf_checkpoint`.

2. **tt-xla** (`remediation/mirothinker_v1_5_30b_gguf-causal_lm-pytorch-v1.5_30B_GGUF-single_device-inference`, commit `2dcc7322f`):
   - `python_package/tt_torch/torch_overrides.py`: Added `_qwen3moe_experts_forward` with device-friendly dense bmm path and monkey-patched `Qwen3MoeExperts.forward`.

3. **tt-xla test config** (commit `4400ae1d6`):
   - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for `mirothinker_v1_5_30b_gguf/causal_lm/pytorch-v1.5_30B_GGUF-single_device-inference`.

## Verification
- pytest exit: FAIL (INTERNAL Error code: 13, hardware-class OOM)
- Hardware:    blackhole-p150b
- Duration:    1854.94s (0:30:54)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 `*/loader.py` files — `_patched_load_gguf_checkpoint` signature fix
- `tt-xla/python_package/tt_torch/torch_overrides.py` — `_qwen3moe_experts_forward` added
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — XFAIL entry added

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4400ae1d60b00a188c27c196b426266bf03bd89f |
| tt-forge-models | 8c3a9d91f647ac1d11f5562a98e43c77c0dac3cf |
