# Remediation Summary: apertus_70b_instruct-causal_lm-pytorch-70B_INSTRUCT_2509-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[apertus_70b_instruct/causal_lm/pytorch-70B_INSTRUCT_2509-single_device-inference]

## Result
XFAIL — 70B model in BF16 (~140 GB) exceeds single-device DRAM (~32 GB); loader bug fixed and test marked KNOWN_FAILURE_XFAIL

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
apertus-mlp-gate-proj-missing, apertus-70b-oom-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
tests/infra/runners/torch_device_runner.py:167: in _safely_put_workload_on_device
    raise AttributeError(
E   AttributeError: 'ApertusMLP' object has no attribute 'gate_proj'
```

After loader fix, test fails with OOM:
```
TT_FATAL: Out of Memory: Not enough space to allocate 704643072 B DRAM buffer
across 8 banks, where each bank needs to store 88080384 B, but bank size is
4273390016 B (allocated: 4112992256 B, free: 160397760 B, largest free block:
71303168 B) (assert.hpp:104)
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
```

## Root cause
Two distinct issues were found:

**1. Loader bug (fixed):** `load_shard_spec` in `apertus_70b_instruct/causal_lm/pytorch/loader.py` referenced `layer.mlp.gate_proj.weight` which does not exist on `ApertusMLP`. The Apertus architecture uses a non-gated MLP (`up_proj → xielu activation → down_proj`) with no `gate_proj`, unlike SwiGLU-style models. This caused `nn.Module.__getattr__` to raise `AttributeError` during device placement (called via `torch_device_runner.py:167`), even for single-device inference.

**2. Hardware capacity (XFAIL):** After the loader fix, the 70B model (70 billion parameters × 2 bytes BF16 = ~140 GB) cannot fit within the single device's available DRAM (~32 GB total across 8 banks of 4 GB each). During execution (`prepareInputTensor`), the runtime attempts to allocate a 704 MB buffer but only ~153 MB per bank (~1.2 GB total) remains free after model weights occupy most of DRAM, and fragmentation means the largest contiguous free block is only ~71 MB per bank. This is a genuine hardware capacity ceiling for single-device inference of a 70B model.

## Fix
**Loader fix** in `tt-forge-models` on branch `remediation/apertus_70b_instruct-causal_lm-pytorch-70B_INSTRUCT_2509-single_device-inference`:
- `apertus_70b_instruct/causal_lm/pytorch/loader.py`: Removed `shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")` from `load_shard_spec`. Added comment clarifying that ApertusMLP is non-gated.

**XFAIL config** in `tt-xla` on branch `remediation/apertus_70b_instruct-causal_lm-pytorch-70B_INSTRUCT_2509-single_device-inference`:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `apertus_70b_instruct/causal_lm/pytorch-70B_INSTRUCT_2509-single_device-inference` with `status: KNOWN_FAILURE_XFAIL` and the OOM error as reason.

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    113.64s (0:01:53)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/apertus_70b_instruct/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ed456908f453fb6c7ae4b3d82107f4cc6b764c79 |
| tt-forge-models | 2f5c5295134d0a913978584eb6f7e9ec38c3e06d |
