# Remediation Summary: llama_3_70b_instruct_abliterated_v3_gguf-causal_lm-pytorch-70B_Instruct_Abliterated_v3_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_70b_instruct_abliterated_v3_gguf/causal_lm/pytorch-70B_Instruct_Abliterated_v3_GGUF-single_device-inference]

## Result
XFAIL — 70B model (Q4_K_M GGUF) dequantizes to ~140 GB BF16, far exceeding single-device DRAM (~96 GB p150b); dequantization alone takes ~69 min causing CI timeout before device OOM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Test exceeded configured timeout and was killed
```

Local reproduction shows the model begins dequantizing 723 GGUF tensors at ~5.73 s/it:
```
Converting and de-quantizing GGUF tensors...:   0%|          | 0/723 [00:00<?, ?it/s]
Converting and de-quantizing GGUF tensors...:   0%|          | 1/723 [00:13<2:43:38, 13.60s/it]
Converting and de-quantizing GGUF tensors...:   0%|          | 2/723 [00:13<1:08:53,  5.73s/it]
```
At that rate, CPU dequantization alone would take ~69 minutes. The CI per-test timeout fires before dequantization completes.

## Root cause
The model `mradermacher/Llama-3-70B-Instruct-abliterated-v3-i1-GGUF` loads via transformers' GGUF loader using `Llama-3-70B-Instruct-abliterated-v3.i1-Q4_K_M.gguf` (Q4_K_M quantization). The HuggingFace GGUF loader dequantizes all 723 tensors to BF16 on CPU before moving weights to device. At ~5.73 s/it for 723 tensors, this takes ~69 minutes — exceeding the CI per-test timeout before the model even reaches the TT device. Even if CPU dequantization completed, a 70B parameter model in BF16 occupies ~140 GB, which far exceeds the ~96 GB DRAM available on a single TT p150b device, causing an OOM during inference. This is a genuine hardware capacity ceiling: the model is too large for single-device inference both in terms of CI time budget (CPU dequantization) and device DRAM.

## Fix
**XFAIL config** (`tt-xla`, branch `remediation/llama_3_70b_instruct_abliterated_v3_gguf-causal_lm-pytorch-70B_Instruct_Abliterated_v3_GGUF-single_device-inference`):
- Added entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  ```yaml
  llama_3_70b_instruct_abliterated_v3_gguf/causal_lm/pytorch-70B_Instruct_Abliterated_v3_GGUF-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    reason: "70B model (Q4_K_M GGUF, 723 tensors) dequantizes to ~140 GB BF16, far exceeding single-device DRAM (~96 GB p150b); dequantization alone takes ~69 min, causing CI timeout before device OOM"
  ```

## Verification
- pytest exit: TIMEOUT (CI-level timeout during CPU dequantization of 723 tensors)
- Hardware:    blackhole-p150b
- Duration:    >69 min (estimated from 5.73 s/it × 723 tensors for CPU dequantization alone)
- Tier A attempts: N/A

## Files changed
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 18e5b54f1a43d24043613e6d525b45be78c1043b |
| tt-forge-models | 1df6fe6061bdaaaca4b929e59897d9abda5dffcf |
