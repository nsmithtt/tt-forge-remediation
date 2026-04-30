# Remediation Summary: gemma_2_9b_it_ko_crypto_translate-causal_lm-pytorch-9B_IT_KO_CRYPTO_TRANSLATE-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_9b_it_ko_crypto_translate/causal_lm/pytorch-gemma_2_9b_it_ko_crypto_translate-single_device-inference]

## Result
XFAIL — model is ~18 GB BF16, exceeds n150 12 GB DRAM; hardware-class failure

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-gemma2-9b-n150-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before fix in tt-xla at 9b2a881cf):
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

After applying the aten.slice OOB fix (already in tt-xla at commits ee94c31a4 and 9b2a881cf),
the test runs but produces PCC=0.183 on n150 due to hardware capacity:
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.18333546980630466. Required: pcc=0.99.

## Root cause
Two issues:

1. aten.slice OOB negative start (already fixed): DynamicSlidingWindowLayer.update() in
   transformers 5.x caches keys/values with self.keys = full_key_states[:, :, -self.sliding_window + 1:, :].
   For Gemma-2-9B with sliding_window=4096 and seq_len=128, the slice start is
   -4095, which is outside XLA's int8 range [-128, 127]. The fix (clamping negative
   slice indices to -size) was applied in tt-xla commits ee94c31a4 (aten.slice.Tensor
   path) and 9b2a881cf (torch.Tensor.__getitem__ path) in torch_overrides.py.

2. Hardware capacity (n150): koalajun/Gemma-2-9b-it-Ko-Crypto-Translate uses the
   gemma2 architecture with 9B parameters (~18 GB BF16). The n150 has 12 GB DRAM, which is
   insufficient. The test runs but produces PCC=0.183 due to memory overflow. The identical
   architecture google/gemma-2-9b-it (gemma/pytorch-2_9B_IT) is already marked
   supported_archs: ["p150"] in the test config for this reason.

## Fix
1. The aten.slice OOB fix is already present in tt-xla at the branch tip (8fb56e2b0):
   - 9b2a881cf: fixes torch.Tensor.__getitem__ slice negative index clamping
   - ee94c31a4: fixes aten.slice.Tensor negative start clamping
   Both changes are in python_package/tt_torch/torch_overrides.py.

2. Test config update: added KNOWN_FAILURE_XFAIL for n150 with arch_overrides marking
   p150 as EXPECTED_PASSING. The remediation branch in tt-xla is:
   remediation/gemma_2_9b_it_ko_crypto_translate-causal_lm-pytorch-9B_IT_KO_CRYPTO_TRANSLATE-single_device-inference
   at commit 0140818b5.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — hardware capacity XFAIL.

## Verification
- pytest exit: FAIL (PCC=0.183 on n150; hardware capacity)
- Hardware:    n150
- Duration:    211.27s (3:31)
- Tier A attempts: N/A
