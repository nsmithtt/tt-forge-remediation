# Remediation Summary: llama_3_2_1b_instruct_quip_w4a16-causal_lm-pytorch-llama_3_2_1b_instruct_quip_w4a16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_1b_instruct_quip_w4a16/causal_lm/pytorch-llama_3_2_1b_instruct_quip_w4a16-single_device-inference]

## Result
SILICON_PASS — missing compressed-tensors dependency added to requirements.txt

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
missing-requirements-compressed-tensors

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.
```

## Root cause
The loader for `nm-testing/Llama-3.2-1B-Instruct-quip-w4a16` had no `requirements.txt`. The model uses the compressed-tensors W4A16 quantization format, which requires the `compressed-tensors` Python package. When transformers 5.x encounters a `quantization_config` using this scheme during `from_pretrained`, it raises `ImportError` if the package is absent. Adding `requirements.txt` with `compressed-tensors` allows the test runner to install the dependency before loading the model.

## Fix
Added `llama_3_2_1b_instruct_quip_w4a16/causal_lm/pytorch/requirements.txt` containing `compressed-tensors` to `tt-forge-models` on branch `remediation/llama_3_2_1b_instruct_quip_w4a16-causal_lm-pytorch-llama_3_2_1b_instruct_quip_w4a16-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    149.29s
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: llama_3_2_1b_instruct_quip_w4a16/causal_lm/pytorch/requirements.txt` (new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 454ce2208624ef44c04546a064e15eb54309567c |
