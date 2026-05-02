# Remediation Summary: insubordinated_plague_parasite_1b_i1_gguf-causal_lm-pytorch-INSUBORDINATED_PLAGUE_PARASITE_1B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[insubordinated_plague_parasite_1b_i1_gguf/causal_lm/pytorch-INSUBORDINATED_PLAGUE_PARASITE_1B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

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
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})
Original traceback:
  ...
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
  File "tt_torch/torch_overrides.py", line 34, in __torch_function__
    return func(*args, **(kwargs or {}))

(The reported failure message `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` was a CI environment issue — gguf was not pre-installed. The loader's `requirements.txt` already contains `gguf>=0.10.0`. With gguf present the test advances past loading and hits the slice OOB bug instead.)

## Root cause
Insubordinated Plague Parasite 1B i1 is a Gemma3-based 1B model with sliding-window attention (sliding_window=512). During inference the KV cache update path computes the sliding-window slice as `full_value_states[:, :, -(sliding_window - 1):, :]` = start=-511. When the prompt is short (seq_len=23), the tensor only has 23 elements in dim 2, so the valid range is [-23, 22]. Python/PyTorch eager semantics silently clamp any start < -size to the beginning of the tensor, but the XLA lazy backend validates bounds strictly and raises "Value out of range". The bug is in tt-xla's `TorchFunctionOverride` — it calls `func(*args, **kwargs)` without pre-clamping the slice indices.

## Fix
Added an `aten.slice.Tensor` intercept at the top of `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`. The intercept clamps start and end to `[-size, size]` for statically-known dimension sizes, matching Python/PyTorch eager semantics before the XLA kernel sees the arguments.

File changed: `python_package/tt_torch/torch_overrides.py`
Branch: `remediation/insubordinated_plague_parasite_1b_i1_gguf-causal_lm-pytorch-INSUBORDINATED_PLAGUE_PARASITE_1B_I1_Q4_K_M_GGUF-single_device-inference` in tt-xla

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    405.73s (0:06:45)
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5e53b8fcfd9b59419ade22f6ebef539aa794a1d5 |
| tt-forge-models | 0499cce45624f1d10aaa1112de80b8c785f792ad |
