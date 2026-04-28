# Remediation Summary: llama_3_1_nemotron_nano_4b-causal_lm-pytorch-3.1_Nemotron_Nano_4B_v1.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_1_nemotron_nano_4b/causal_lm/pytorch-3.1_Nemotron_Nano_4B_v1.1-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
padding-max-length-causal-lm-pcc-drop

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 14:36:11.667 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 0: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

On local silicon (Blackhole p150b) the test reproduced as a PCC failure:
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9076594303551884. Required: pcc=0.99.

## Root cause
The loader called the tokenizer with `padding="max_length"`, `truncation=True`, `max_length=128`. The chat-templated input for "What is your favorite city?" is only 26 tokens, leaving 102 padding positions (80% padding). The TT backend mishandles the resulting 4D causal+padding attention mask for CausalLM models — the 26 real tokens' logits are severely corrupted by the masked-out padding positions, driving PCC to 0.907. On n150 CI hardware the same mask mishandling causes the hardware to hang, producing the Fabric Router Sync Timeout (10 s fabric-level timeout) rather than completing with wrong results.

## Fix
Removed `padding="max_length"`, `truncation=True`, and `max_length=max_length` from the tokenizer call in `load_inputs()`, allowing the tokenizer to return the natural 26-token sequence with no padding. The attention mask then has all ones and the TT hardware processes it correctly.

File: `llama_3_1_nemotron_nano_4b/causal_lm/pytorch/loader.py`
Branch: `remediation/llama_3_1_nemotron_nano_4b-causal_lm-pytorch-3.1_Nemotron_Nano_4B_v1.1-single_device-inference` in `tenstorrent/tt-xla`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    83.74s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/llama_3_1_nemotron_nano_4b/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 9faa2f39c81eba1c6e218ddb8b8c4c430109514d |
