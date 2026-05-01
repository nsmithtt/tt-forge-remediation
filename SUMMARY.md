# Remediation Summary: huihui_qwen3_next_80b_a3b_thinking_abliterated_gguf-causal_lm-pytorch-80B_A3B_Thinking_abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_next_80b_a3b_thinking_abliterated_gguf/causal_lm/pytorch-80B_A3B_Thinking_abliterated_GGUF-single_device-inference]

## Result
XFAIL — 80B model dequantises to ~160 GB BF16 at load time, exceeding single-device DRAM (p150b: 24 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-qwen3next-arch-not-registered-80b-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
Two bugs coexist:

**Loader bug**: The GGUF file uses `general.architecture = 'qwen3next'` (no underscore) but
`qwen3next` is absent from `GGUF_SUPPORTED_ARCHITECTURES` in transformers 5.2.0 (which only
has `qwen3` and `qwen3_moe`). This would cause `ValueError: GGUF model with architecture
qwen3next is not supported yet.` once the file is available. The transformers model class
`Qwen3NextForCausalLM` (model_type `qwen3_next`) is present in transformers 5.2.0 — only the
GGUF architecture string registration is missing.

**Hardware-class capacity**: The GGUF Q4_K_M file is 48.4 GB (80B total params at ~4.85
bits/weight). The transformers GGUF loader dequantises all weights to BF16 at load time
(80B × 2 bytes = 160 GB), far exceeding any single TT device DRAM (n150: 12 GB, p150b:
24 GB). The CI timeout was caused by the 48.4 GB download never completing within the
test time limit.

## Fix
**Loader fix** in `tt_forge_models/huihui_qwen3_next_80b_a3b_thinking_abliterated_gguf/causal_lm/pytorch/loader.py`:
- Registered `qwen3next` in `GGUF_SUPPORTED_ARCHITECTURES` at import time.
- Added `GGUF_TO_TRANSFORMERS_MAPPING['config']['qwen3next']` with field mappings derived
  from the GGUF KV metadata (block_count, embedding_length, attention.*, rope.*, expert_*,
  ssm.conv_kernel → linear_conv_kernel_dim).
- Added `GGUF_TO_FAST_CONVERTERS['qwen3next']` pointing to the `qwen3` tokenizer converter
  (Qwen3Next uses Qwen2Tokenizer per transformers tokenization_auto.py).
- Wrapped `load_gguf_checkpoint` to remap `model_type: 'qwen3next'` → `'qwen3_next'` so
  `AutoModelForCausalLM` resolves to `Qwen3NextForCausalLM`. Wrapper uses `**kwargs` to
  handle the `model_to_load=None` parameter added in transformers 5.x.
- Added `config._experts_implementation = "batched_mm"` before `from_pretrained` to avoid
  `histc-on-int` failure under TT's XLA device (device.type == "xla" triggers int histc in
  the default `grouped_mm` experts path).
- Added `requirements.txt` with `gguf>=0.10.0`.

**Test config** in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
- Added `KNOWN_FAILURE_XFAIL` entry for this test.

## Verification
- pytest exit: TIMEOUT (cannot reproduce: 48.4 GB GGUF file unavailable; loader fix validated syntactically)
- Hardware:    not-run
- Duration:    not-run
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/huihui_qwen3_next_80b_a3b_thinking_abliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/huihui_qwen3_next_80b_a3b_thinking_abliterated_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bf72241a28dc1f0379dd07f2602ff2ad08463c17 |
| tt-forge-models | 557202337483cde69ee7743cca7547592df20d2d |
