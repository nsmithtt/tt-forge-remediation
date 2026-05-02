# Remediation Summary: mozilla_ai_gemma_2_2b_it_llamafile-causal_lm-pytorch-gemma-2-2b-it-llamafile-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mozilla_ai_gemma_2_2b_it_llamafile/causal_lm/pytorch-gemma-2-2b-it-llamafile-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original error (reported): RuntimeError: Value out of range (expected to be in range of [-17, 16], but got -4095)

Actual first failure observed:
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a tokenizers library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.

After loader fix, second failure:
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

After context manager fix, third failure (original compiler error):
RuntimeError: Value out of range (expected to be in range of [-24, 23], but got -4095)
While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

## Root cause
Three bugs, each blocking the next:

1. Loader bug (HF repo mismatch): mozilla-ai/gemma-2-2b-it-llamafile is a repository of llamafile executables (GGUF models wrapped in shell scripts), not standard HuggingFace model files. AutoTokenizer.from_pretrained fails because there is no tokenizer.json or tokenizer_config.json. Fix: redirect to bartowski/gemma-2-2b-it-GGUF with gguf_file="gemma-2-2b-it-Q4_K_M.gguf".

2. Loader bug (GGUF load_gguf_checkpoint pollution): 26+ other loaders in tt-forge-models patch transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint at module import time with versions missing the model_to_load kwarg added in transformers 5.x. During pytest collection all loader.py files are imported, so these patches persist when our test runs. Fix: _use_real_load_gguf_checkpoint() context manager using gc.get_objects() to find and restore the original function.

3. Compiler frontend bug (aten.slice.Tensor OOB start): Gemma 2 uses sliding window attention with sliding_window=4096. The KV cache update computes full_value_states[:, :, -sliding_window+1:, :] = start=-4095. With seq_len=24, XLA validates against [-24, 23] and rejects -4095. Fix: clamp in TorchFunctionOverride.__torch_function__ for func is torch.ops.aten.slice.Tensor.

## Fix
Fix 1 - Loader (tt_forge_models):
- File: mozilla_ai_gemma_2_2b_it_llamafile/causal_lm/pytorch/loader.py
- Changed pretrained_model_name from mozilla-ai/gemma-2-2b-it-llamafile to bartowski/gemma-2-2b-it-GGUF
- Added _GGUF_FILE = "gemma-2-2b-it-Q4_K_M.gguf" class constant
- Added gguf_file=self._GGUF_FILE to from_pretrained calls
- Added _find_real_load_gguf_checkpoint() (gc scan) + _use_real_load_gguf_checkpoint() context manager
- Branch: remediation/mozilla_ai_gemma_2_2b_it_llamafile-causal_lm-pytorch-gemma-2-2b-it-llamafile-single_device-inference on tenstorrent/tt-forge-models

Fix 2 - Compiler frontend (tt-xla):
- File: python_package/tt_torch/torch_overrides.py
- Added slice start/end clamping in TorchFunctionOverride.__torch_function__ for func is torch.ops.aten.slice.Tensor
- Clamps args[2] (start) and args[3] (end) to [-dim_size, dim_size] before dispatch to XLA
- Branch: remediation/mozilla_ai_gemma_2_2b_it_llamafile-causal_lm-pytorch-gemma-2-2b-it-llamafile-single_device-inference on tenstorrent/tt-xla

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    324.46s (0:05:24)
- Tier A attempts: 1

## Files changed
- mozilla_ai_gemma_2_2b_it_llamafile/causal_lm/pytorch/loader.py (tt-forge-models)
- python_package/tt_torch/torch_overrides.py (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fda469fc7ab7c285f59ee98dc6e693045719cef7 |
| tt-forge-models | 2b5b4054d07278cc778b83a8d10b862ed57c56b5 |
