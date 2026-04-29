# Remediation Summary: gemma3_12b_cybersecurity_gguf-causal_lm-pytorch-12B_cybersecurity_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_12b_cybersecurity_gguf/causal_lm/pytorch-12B_cybersecurity_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-33, 32], but got -1023)

## Root cause
Two independent bugs blocked this test:

1. **Loader bug** (`_patched_load_gguf_checkpoint` narrow signature): Several GGUF loaders
   (26 qwen3.5/gpt-oss loaders) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
   at import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 now
   calls it with an additional `model_to_load=dummy_model` keyword argument. When pytest collects test
   parametrization it imports all loader modules, so these patches persist into the gemma3 test run.
   When `AutoModelForCausalLM.from_pretrained` internally calls `load_gguf_checkpoint(..., model_to_load=...)`
   it hits the patched version and raises `TypeError`.

2. **tt-xla frontend bug** (XLA lazy slice out-of-bounds): Gemma 3's sliding-window attention
   KV cache uses negative slice indices beyond `[-size, size-1]` (e.g. start=-1023 on a dim-33
   cache). PyTorch eager silently clamps such indices; the XLA lazy backend raises
   `RuntimeError: Value out of range`. The fix is to pre-clamp these indices in
   `TorchFunctionOverride.__torch_function__` before they reach the lazy backend.

   Also: the gemma3_12b_cybersecurity_gguf loader was missing a `requirements.txt` declaring
   `gguf>=0.10.0`, which could cause an `ImportError` on fresh environments.

## Fix
**Loader fix** (`tt_forge_models`, remediation branch `remediation/gemma3_12b_cybersecurity_gguf-causal_lm-pytorch-12B_cybersecurity_GGUF-single_device-inference`):
- `gemma3_12b_cybersecurity_gguf/causal_lm/pytorch/requirements.txt`: added `gguf>=0.10.0`
- 26 GGUF loader files: changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`
  to `def _patched_load_gguf_checkpoint(*args, **kwargs):` and updated the inner call from
  `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to
  `_orig_load_gguf_checkpoint(*args, **kwargs)` (cherry-picked commits 09cb8b3cc7 and 19ea50ddf7
  from `origin/ip-172-31-23-5-tt-xla-dev/ubuntu/hf-bringup-35`)

**tt-xla fix** (`tt-xla`, remediation branch `remediation/gemma3_12b_cybersecurity_gguf-causal_lm-pytorch-12B_cybersecurity_GGUF-single_device-inference`):
- `python_package/tt_torch/torch_overrides.py`: added intercept for `func is torch.ops.aten.slice.Tensor`
  in `TorchFunctionOverride.__torch_function__`; clamps `start` and `end` to `[-size, size]` when
  they fall below `-size` and the dimension size is a known int.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    714.48s (0:11:54)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gemma3_12b_cybersecurity_gguf/causal_lm/pytorch/requirements.txt` (new file)
- `tt_forge_models/<26 qwen3.5/gpt-oss loaders>/causal_lm/pytorch/loader.py` (`_patched_load_gguf_checkpoint` signature)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (slice index clamping)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f0b6758953d8ed64dc5e8bb36c898a36dd67a525 |
| tt-forge-models | 7f21734f933ce526fed4c30724f4d75f8751e744 |
