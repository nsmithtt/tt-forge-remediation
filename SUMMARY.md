# Remediation Summary: deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf-causal_lm-pytorch-DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch-DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — complex tensor operations (torch.polar → complex64 mul) crash in TT XLA dispatch

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
aten-complex-mul-crash-tt-xla

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After installing gguf (current environment), failure becomes:
```
KeyError: 'deepseek_v2'
```
at `transformers/integrations/ggml.py:787: converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)` — the GGUF tokenizer architecture `deepseek_v2` is not registered in `GGUF_TO_FAST_CONVERTERS`, and a global patch from the iquest_coder loader intercepts the call first.

After applying the loader fix (remediation branch `remediation/deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf-causal_lm-pytorch-DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference` in tt_forge_models), the model loads successfully but the test fails at inference with:
```
While executing %mul : [num_users=27] = call_function[target=torch.ops.aten.mul.Tensor](args = (%polar, 1.0), kwargs = {})
Original traceback:
  .../deepseek_v2/modeling_deepseek_v2.py:229: freqs_cis = freqs_cis * self.attention_scaling
```
with a C++ crash backtrace from `at::_ops::mul_Tensor::call` inside the TorchFunctionMode `__torch_function__` override.

## Root cause
Two separate bugs:

**Bug 1 (loader, fixed):** Transformers 5.x does not ship `deepseek_v2`/`deepseek2` in `GGUF_TO_FAST_CONVERTERS`, `GGUF_SUPPORTED_ARCHITECTURES`, or `GGUF_TO_TRANSFORMERS_MAPPING`. The GGUF file uses `deepseek2` as its raw architecture key, and `deepseek_v2` is the HF model_type. Additionally:
- The iquest_coder_v1_40b_instruct_gguf loader globally patches `convert_gguf_tokenizer` at import time, causing interference across loader sessions.
- The GGUF config incorrectly maps `head_count_kv=1` (the MLA compressed rank) as `num_key_value_heads`, causing incorrect 16× GQA expansion; `key_length_mla` stores the full key dim (nope+rope) not just nope; and `q_lora_rank` defaults to 1536 but this GGUF has a fused Q matrix.
- The transformers 5.x `load_gguf_checkpoint` requires a `model_to_load` keyword argument that earlier loaders drop.
- `gguf>=0.10.0` was missing from `requirements.txt`.

**Bug 2 (compiler, unfixed):** DeepSeek-V2's RoPE implementation uses complex tensor arithmetic: `torch.polar(ones, freqs)` creates a `complex64` tensor, `freqs_cis * attention_scaling` applies scalar mul, `view_as_complex` / `view_as_real` are used in `apply_rotary_emb`. The TT XLA backend's `__torch_function__` override calls `func(*args)` for `aten.mul.Tensor` on a `complex64` XLA tensor, which crashes in the XLA lazy dispatch path — `at::_ops::mul_Tensor::call` does not support complex64 dtype on TT XLA. The crash occurs before the MLIR pipeline (`stablehlo-complex-math-expander` + `ComplexDataTypeConversion`) can process the computation.

## Fix
**Bug 1 (loader fix):** Implemented in `tt_forge_models` on branch `remediation/deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf-causal_lm-pytorch-DEEPSEEK_CODER_V2_LITE_13B_INSTRUCT_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference` (commit `7f054b135d9175adbad6841b0a4d8941d47981b8`):
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch/loader.py` — complete rewrite: registers `deepseek2` in GGUF metadata maps, adds `deepseek2`/`deepseek_v2` to `GGUF_TO_FAST_CONVERTERS` with `GGUFLlamaConverter`, patches `get_gguf_hf_weights_map` to translate `deepseek_v2`→`deepseek2`, provides context manager that installs a correct `load_gguf_checkpoint` wrapper accepting `model_to_load`, corrects `num_key_value_heads`, `qk_nope_head_dim`, and `q_lora_rank` in the GGUF config.
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch/requirements.txt` — adds `gguf>=0.10.0`.

**Bug 2 (proposed fix):** In `tt-xla/python_package/tt_torch/torch_overrides.py`, the `TorchFunctionOverride.__torch_function__` should intercept `aten.mul.Tensor` (and related ops: `aten.polar`, `aten.view_as_complex`, `aten.view_as_real`) when the input tensor has complex dtype, and decompose them into real-valued equivalents before calling `func`. Specifically:
- `aten.mul.Tensor(complex, scalar)` → `view_as_complex(view_as_real(complex) * scalar)`
- `aten.mul.Tensor(complex, complex)` → full complex multiplication formula `(a*c - b*d) + i*(a*d + b*c)`
- Other complex ops would need similar handling

## Tier B justification
cross-cutting: Proper fix for complex tensor operations in the TT XLA backend requires intercepting and correctly decomposing at least 5 aten ops (`polar`, `mul` complex×scalar, `mul` complex×complex, `view_as_complex`, `view_as_real`) in the `__torch_function__` override. The complex × complex case requires a non-trivial 4-multiplication decomposition. Additionally, there may be other complex-dtype aten ops used by DeepSeek-V2 or other models that would also need to be handled. The fix cannot be confidently scoped to a single pattern in a single function without comprehensive testing of the complex tensor path.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    929.49s (0:15:29) — model loading dominates; test confirmed fails at compiler stage after successful model load
- Tier A attempts: 0

## Files changed
In `tt_forge_models` (`remediation/deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf-...` branch):
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/__init__.py` (new)
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/__init__.py` (new)
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch/__init__.py` (new)
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch/loader.py` (new, complete implementation)
- `deepseek_coder_v2_lite_13b_instruct_sft_s1k_i1_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7f054b135d9175adbad6841b0a4d8941d47981b8 |
