# Remediation Summary: huihui_mistral_small_3_2_24b_instruct_abliterated_v2_gguf-causal_lm-pytorch-24B_INSTRUCT_ABLITERATED_LLAMACPPFIXED_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_mistral_small_3_2_24b_instruct_abliterated_v2_gguf/causal_lm/pytorch-24B_INSTRUCT_ABLITERATED_LLAMACPPFIXED_GGUF-single_device-inference]

## Result
XFAIL — 24B GGUF model dequantized to BF16 (~48 GB) exceeds p150b device DRAM (~32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements-txt

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After loader fix (requirements.txt added), the terminal failure is device OOM:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
where each bank needs to store 41943040 B, but bank size is 4273390016 B
(allocated: 4196977728 B, free: 76412288 B, largest free block: 37030336 B)
```

## Root cause
Two issues:

1. **Loader bug**: `requirements.txt` with `gguf>=0.10.0` was missing from the loader directory. When `gguf` is absent from the venv, transformers raises `ImportError` before any model loading occurs.

2. **Hardware capacity ceiling**: The 24B GGUF Q4_K_M model is dequantized to BF16 for TT device inference. At BF16 (2 bytes/param), 24B parameters = ~48 GB — this exceeds the p150b's ~32 GB device DRAM (8 banks × ~4 GB/bank). After filling ~3.91 GB per bank with model weights, there is insufficient contiguous memory for activation buffers during inference.

## Fix
1. **tt_forge_models** (`huihui_mistral_small_3_2_24b_instruct_abliterated_v2_gguf/causal_lm/pytorch/`):
   - Added `requirements.txt` containing `gguf>=0.10.0`
   - Fixed `load_inputs()` to guard `apply_chat_template` with `if self.tokenizer.chat_template is not None:`, falling back to plain sample text

2. **tt-xla** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
   - Added `KNOWN_FAILURE_XFAIL` entry for the 24B LLAMACPPFIXED GGUF variant with the OOM reason

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL (OOM on device after loader fix)
- Hardware:    blackhole-p150b
- Duration:    781.50s (0:13:01)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huihui_mistral_small_3_2_24b_instruct_abliterated_v2_gguf/causal_lm/pytorch/requirements.txt` (added)
- `tt-xla/third_party/tt_forge_models/huihui_mistral_small_3_2_24b_instruct_abliterated_v2_gguf/causal_lm/pytorch/loader.py` (apply_chat_template guard)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (KNOWN_FAILURE_XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 52b0e0e61dbb59f39323507d98dc44edcd1a01ca |
| tt-forge-models | 8df7e30524e5e6f5a2ef0bdd9453f36a11e0f2e2 |
