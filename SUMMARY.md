# Remediation Summary: aratako_qwen3_30b_a3b_nsfw_jp-causal_lm-pytorch-30B_A3B_NSFW_JP-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aratako_qwen3_30b_a3b_nsfw_jp/causal_lm/pytorch-30B_A3B_NSFW_JP-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B-NSFW-JP requires ~60 GB device DRAM in BF16; exceeds single p300c capacity (~24 GB per chip)

## Stack layer
tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
qwen3moe-nonzero-dynamic-shape-mhlo-stablehlo-fail

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Check failed: status.ok(): MHLO -> StableHLO conversion failed.

## Root cause
`Qwen3MoeExperts.forward` in transformers uses a data-dependent `nonzero()`
call followed by a `for expert_idx in expert_hit` loop and `index_add_` to
scatter per-expert outputs. These dynamic-shape operations cause `torch.compile`
with the XLA backend to hang indefinitely during graph tracing (or, with
bringup's timeout mechanism, to surface as an MHLO → StableHLO conversion
failure). The transformers `@use_experts_implementation` decorator dispatches
to `grouped_mm_experts_forward` which uses `torch._grouped_mm` — also not
XLA-aware — depending on the installed version.

Separately, the full 30B model in BF16 requires approximately 60 GB of device
DRAM, which exceeds the ~24 GB available on a single p300c Blackhole chip. The
model cannot be loaded to a single device regardless of the compilation fix.

## Fix
**tt-xla (compiler frontend):** Added `_qwen3_moe_experts_forward` monkey-patch
in `python_package/tt_torch/torch_overrides.py`. The device path replaces the
dynamic-shape per-expert loop with a static-shape dense bmm:
- scatter routing weights into a full `[T, E]` matrix
- tile hidden states to `[E, T, H]`
- `torch.bmm` with transposed `gate_up_proj` and `down_proj` weight tensors
- weighted sum over the expert dimension

CPU path retains the per-expert loop for use as a PCC golden reference.

**tt-xla (test config):** Added `KNOWN_FAILURE_XFAIL` entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
because the 30B model exceeds single-device hardware capacity even after the
compilation fix. This is a hardware-class ceiling, not a compiler bug.

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    82.32s (0:01:22)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `_qwen3_moe_experts_forward` and monkey-patch of `Qwen3MoeExperts.forward`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` for this test

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bdbb94760935b57032e21b4b672386ebf4a1a729 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
