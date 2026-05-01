# Remediation Summary: li101_qwen3_5_35b_a3b_uncensored_aggressive_safetensors/causal_lm/pytorch-35B_A3B_Uncensored_Aggressive_safetensors-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[li101_qwen3_5_35b_a3b_uncensored_aggressive_safetensors/causal_lm/pytorch-35B_A3B_Uncensored_Aggressive_safetensors-single_device-inference]

## Result
XFAIL — 35B BF16 model (~70 GB) exceeds single-device DRAM (p150b: 32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-35b-bf16-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: expected data_ptr to be aligned to 16 bytes

(The originally-reported `MHLO -> StableHLO conversion failed` is the same root issue on a different run; see Root cause below.)

## Root cause
Two separate bugs prevent this test from passing, with a hardware-capacity ceiling being the definitive reason for the XFAIL classification.

**Bug 1 (loader, unfixed in this branch):** The safetensors checkpoint stores expert weights at a file offset that is 8-byte-aligned but not 16-byte-aligned (data section starts at byte 696 from the beginning of the first shard, i.e., 696 % 16 = 8). Transformers 5.x selects `grouped_mm` as the default MoE dispatch when PyTorch ≥ 2.9 is present (`is_grouped_mm_available()` returns `True` for torch 2.9.1+cpu). `grouped_mm_experts_forward` calls `torch._grouped_mm`, which enforces 16-byte alignment of all input `data_ptr` values. The mmap-backed safetensors tensor violates this, causing `RuntimeError: expected data_ptr to be aligned to 16 bytes` during the CPU reference forward pass — before TT compilation is ever reached.

A fix for this specific loader was previously committed (`d9d5427d83` on the `tenstorrent/tt-forge-models` repository) by switching `_experts_implementation` to `"batched_mm"` after `from_pretrained`. That commit is **not** an ancestor of the current configured branch (`hf-bringup-49`), so this report re-encounters the same alignment failure.

**Bug 2 (tt-mlir, Tier B, pre-existing):** After fixing Bug 1, the model's `linear_attention` layers use GatedDeltaNet (`torch_chunk_gated_delta_rule`), whose Python scan loop unrolls into a pathologically large StableHLO graph during XLA tracing. A prior report (`report/li101_...`, skill version 6) documented this causing 90+ GB host RAM consumption and a 57-minute compilation hang on n150 hardware, resulting in `MHLO -> StableHLO conversion failed`. This Tier B bug remains unfixed.

**Hardware capacity ceiling (primary):** Even with both bugs fixed, the full model (~35B BF16 parameters ≈ 70 GB) cannot fit in the 32 GB DRAM of a single p150b device. This is the definitive reason for `KNOWN_FAILURE_XFAIL`.

## Fix
No fix is applied. The test config entry for this model is marked `KNOWN_FAILURE_XFAIL` with reason: "35B BF16 model (~70 GB) exceeds single-device DRAM (p150b: 32 GB)".

File changed: `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

The previously committed loader fix (`d9d5427d83` — switch to `batched_mm` after `from_pretrained`) addresses Bug 1 but is not included in the current branch; it is tracked separately.

## Verification
- pytest exit: FAIL (CPU reference pass fails with alignment error; TT device never reached)
- Hardware:    blackhole-p150b
- Duration:    24.93s (failed in CPU reference pass)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add `KNOWN_FAILURE_XFAIL` entry for this model

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 14819a113a86ed00a2379cb272cbcc90f52a3604 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
