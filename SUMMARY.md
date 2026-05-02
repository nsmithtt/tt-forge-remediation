# Remediation Summary: kirillr_qwq_32b_preview_awq-causal_lm-pytorch-QwQ_32B_Preview_AWQ-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kirillr_qwq_32b_preview_awq/causal_lm/pytorch-QwQ_32B_Preview_AWQ-single_device-inference]

## Result
XFAIL — 32B model dequantized to BF16 (~64 GB) exceeds single-device DRAM (~34 GB available on p150b)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
awq-gptqmodel-transitive-dep-env-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure: `ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with pip install gptqmodel`

After loader fix, terminal failure: `RuntimeError: TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 283115520 B DRAM buffer across 8 banks, where each bank needs to store 35389440 B, but bank size is 4273390016 B (allocated: 4178216960 B, free: 95173056 B, largest free block: 28835840 B)`

## Root cause
Two issues:

1. **Loader bug**: transformers 5.x requires `gptqmodel` to load AWQ quantized models. The loader had no `requirements.txt` or `requirements.nodeps.txt`, so gptqmodel was never installed. Additionally, gptqmodel 7.0.0 has multiple transitive dependencies (pypcre, logbar, device-smi, defuser, tokenicer) that must be installed with `--no-deps` to avoid polluting the environment.

2. **Hardware-class capacity**: After the loader fix, gptqmodel dequantizes the INT4 AWQ weights to BF16 before any forward pass, expanding the model from ~4.5 GB INT4 to ~64 GB BF16. The QwQ-32B model has hidden_size=5120, 64 layers, intermediate_size=27648. At BF16 this is approximately 64 GB total. The p150b device reports only ~34 GB of available DRAM across 8 banks (bank_size=4.27 GB per bank × 8 = 34.2 GB), which is insufficient for a 64 GB model. This is consistent with `qwen_2/causal_lm/pytorch-Qwq_32B-single_device-inference` which is already `EXCLUDE_MODEL` with reason "Too large for single chip, run as tensor_parallel instead."

## Fix
1. **Loader fix** (tt-forge-models, `kirillr_qwq_32b_preview_awq/causal_lm/pytorch/`):
   - Added `requirements.txt` (SPDX header only, activates requirements manager)
   - Added `requirements.nodeps.txt` with gptqmodel and all transitive deps installed with `--no-deps`
   - Added `_dequantize_awq_layers()` to replace AWQ quantized linear layers with standard `nn.Linear` (BF16) before any forward pass
   - Called `_dequantize_awq_layers()` in `load_model()` immediately after `from_pretrained()`

2. **Test config** (tt-xla, `tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
   - Added `KNOWN_FAILURE_XFAIL` entry for this test with reason explaining the hardware-class OOM

## Verification
- pytest exit: FAIL (OOM at 573.57s) — test config updated to KNOWN_FAILURE_XFAIL so future runs will report xfail
- Hardware:    blackhole-p150b
- Duration:    573.57s (0:09:33)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: kirillr_qwq_32b_preview_awq/causal_lm/pytorch/loader.py`
- `tt-forge-models: kirillr_qwq_32b_preview_awq/causal_lm/pytorch/requirements.txt` (new)
- `tt-forge-models: kirillr_qwq_32b_preview_awq/causal_lm/pytorch/requirements.nodeps.txt` (new)
- `tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b0b40fffc02acba4e4630ef0f1c5c60db553d6ec |
| tt-forge-models | 0dedd6c47e6cd323bf5c2b983b8d7db2f36d1884 |
