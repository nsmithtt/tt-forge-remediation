# Remediation Summary: llama_3_1_nemotron_70b_instruct_hf_gguf-causal_lm-pytorch-Bartowski_3.1_Nemotron_70B_Instruct_HF_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_1_nemotron_70b_instruct_hf_gguf/causal_lm/pytorch-Bartowski_3.1_Nemotron_70B_Instruct_HF_Q4_K_M-single_device-inference]

## Result
XFAIL — Llama 3.1 Nemotron 70B Q4_K_M (~40 GB) exceeds p150b single-device DRAM (32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
llama-70b-q4km-dram-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
CI failure: `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

After adding `requirements.txt`, confirmed hardware OOM on device:
`RuntimeError: TT_FATAL @ tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false`

## Root cause
Two issues:

1. **Loader (missing dependency):** `gguf>=0.10.0` was absent from `requirements.txt`. The
   RequirementsManager does not install it, so `transformers` raises ImportError when
   attempting to parse the GGUF checkpoint.

2. **Hardware capacity:** Llama 3.1 70B in Q4_K_M format is approximately 40 GB on disk.
   When loaded via `from_pretrained`, transformers dequantizes the weights to BF16,
   producing a ~141 GB in-memory model. The p150b Blackhole device has 32 GB DRAM.
   Even the quantized form (~40 GB) exceeds the device limit, making single-device
   inference impossible. Confirmed by `TT_FATAL` in `bank_manager.cpp:439` when the
   model attempts to allocate DRAM buffers on device — the same hardware-class failure
   seen in BrownLoafers 70B and Llama-MiraiFanfare-2-3.3-70B.

## Fix
- **Loader fix (tt_forge_models):** Added `gguf>=0.10.0` to
  `llama_3_1_nemotron_70b_instruct_hf_gguf/causal_lm/pytorch/requirements.txt`.

- **Test config (tt-xla):** Added `KNOWN_FAILURE_XFAIL` entry in
  `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  ```yaml
  llama_3_1_nemotron_70b_instruct_hf_gguf/causal_lm/pytorch-Bartowski_3.1_Nemotron_70B_Instruct_HF_Q4_K_M-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "Hardware capacity: Llama 3.1 70B Q4_K_M (~40 GB) exceeds p150b single-device DRAM (32 GB)"
  ```

Note: The `_patched_load_gguf_checkpoint` cross-loader clobbering (model_to_load TypeError)
was already resolved in the configured branch HEAD; no additional fix was needed.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    ~220s (0:03:40) — model loaded from 40 GB cache then hit DRAM OOM on device
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/llama_3_1_nemotron_70b_instruct_hf_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 329e55f88903a36d3ce021f0a100be52362a5eb3 |
| tt-forge-models | 905ecdb401e16e5c7a63983ce307469148b7869e |
