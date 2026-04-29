# Remediation Summary: andy_4_1_i1_gguf-causal_lm-pytorch-4_1_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[andy_4_1_i1_gguf/causal_lm/pytorch-4_1_I1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-qwen3vl-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture qwen3vl is not supported yet.
```

The original CI failure message was `raise AttributeError(` — this is the same
root cause manifesting further down the stack when the unrecognized model_type
propagates to `AutoConfig.for_model`.

## Root cause
Andy-4.1 is a text-only fine-tune of Qwen3-VL. Its GGUF
(`mradermacher/Andy-4.1-i1-GGUF`, file `Andy-4.1.i1-Q4_K_M.gguf`) carries
`general.architecture = qwen3vl`. The GGUF contains no visual encoder
tensors (verified by inspecting all 530 tensor names), so the
language-decoder weights are identical to a plain Qwen3 model.

Two interlocking loader bugs caused the failure:

1. **Missing GGUF architecture registration** — transformers 5.2.0's
   `GGUF_CONFIG_MAPPING` supports `qwen3` and `qwen3_moe` but not `qwen3vl`.
   `load_gguf_checkpoint` raises `ValueError` as soon as it sees the
   architecture string.

2. **Broken GGUF patching chain** — 26+ other loaders patch
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import
   time. Their wrappers omit the `model_to_load` kwarg added in transformers
   5.x. The outermost wrapper in a full pytest session rejects the kwarg with
   `TypeError`, which surfaces as `AttributeError` in the CI logs. Even if
   architecture (1) were fixed globally, the model-loading call would still
   fail because `AutoModelForCausalLM.from_pretrained` passes
   `model_to_load=dummy_model` to `load_gguf_checkpoint`, and `model_to_load`
   is required (not optional) when `return_tensors=True`.

Both bugs live entirely in the loader layer.

## Fix
Changes in `third_party/tt_forge_models/andy_4_1_i1_gguf/causal_lm/pytorch/loader.py`:

**Fix 1 — register qwen3vl at import time:**
```python
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations import GGUF_CONFIG_MAPPING as _GGUF_CONFIG_MAPPING

if "qwen3vl" not in _GGUF_CONFIG_MAPPING and "qwen3" in _GGUF_CONFIG_MAPPING:
    _GGUF_CONFIG_MAPPING["qwen3vl"] = dict(_GGUF_CONFIG_MAPPING["qwen3"])
    if "qwen3vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
```
This lets `load_gguf_checkpoint` parse the GGUF metadata without raising.

**Fix 2 — install a context-local wrapper around each HF call:**

A `_find_real_load_gguf_checkpoint()` helper walks the closure chain to
recover the original transformers function (bypassing the 26-loader chain
whose outermost member omits `model_to_load`). A context manager
`_qwen3vl_gguf_context()` installs a wrapper as outermost at all module-level
binding sites (`modeling_gguf_pytorch_utils`, `configuration_utils`,
`tokenization_utils_tokenizers`, `tokenization_auto`) that:

- Accepts `model_to_load` and forwards it to the real function.
- Remaps `model_type "qwen3vl"` → `"qwen3"` in the returned config so
  `AutoModelForCausalLM` resolves to `Qwen3ForCausalLM`.

The context manager is entered in `_load_tokenizer()`, `load_model()`, and
`load_config()`, and is exited (restoring all binding sites) before returning.

`tt_forge_models` commit `fbcdd81f3e` on branch
`remediation/andy_4_1_i1_gguf-causal_lm-pytorch-4_1_I1_GGUF-single_device-inference`.

`tt-xla` commit `a5fbe5a79` on branch
`remediation/andy_4_1_i1_gguf-causal_lm-pytorch-4_1_I1_GGUF-single_device-inference`
(advances `third_party/tt_forge_models` pointer to `fbcdd81f3e`).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    404.20s (0:06:44)
- Tier A attempts: N/A

## Files changed
- `andy_4_1_i1_gguf/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | a5fbe5a79bee5d3b508405708906ff862889a3f0   |
| tt-forge-models | fbcdd81f3e8d640d2667bc90f32a455436ed479a   |
