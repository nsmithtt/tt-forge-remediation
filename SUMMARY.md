# Remediation Summary

## Test

```
tests/runner/test_models.py::test_all_models_torch[apertus_8b_instruct_2509_gguf/causal_lm/pytorch-Apertus_8B_Instruct_2509_Q4_K_M-single_device-inference]
```

## Result: SILICON_PASS

The test passes on the `arch-c-36-tt-xla-dev/nsmith/hf-bringup-47` branch of
`tt-forge-models`.

## Root Cause

The GGUF file for `unsloth/Apertus-8B-Instruct-2509-GGUF` declares
`general.architecture = "apertus"` but `transformers` (v5.2.0) does not
register `apertus` in its GGUF loader tables
(`GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`), even
though it has a full `ApertusForCausalLM` model class.

Running the test on the base branch therefore raised:

```
ValueError: GGUF model with architecture apertus is not supported yet.
```

The originally reported failure (`pcc=nan`) was the downstream symptom
reported on a prior run; the underlying error is the missing GGUF
architecture registration.

## Fix

Commit `215d1080a2` on branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-47` in
`tt-forge-models` updates
`apertus_8b_instruct_2509_gguf/causal_lm/pytorch/loader.py` to register the
`apertus` architecture at module-import time via `_patch_apertus_gguf_support()`:

- Appends `"apertus"` to `GGUF_SUPPORTED_ARCHITECTURES`
- Copies the `llama` config-field mapping into
  `GGUF_TO_TRANSFORMERS_MAPPING["config"]["apertus"]` (Apertus uses
  identical GGUF field names: `block_count`, `context_length`, etc.)
- Aliases the `llama` fast tokenizer converter for `apertus`

No changes were required in `tt-xla`, `tt-mlir`, or `tt-metal`.

## Submodule Hashes

| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-xla          | `94362e631171473c01993b3e216b6ae8ebb93ab8` |
| tt-mlir         | `553c0632b353f8ac457aba0d01a460a5e0f5b5ee` |
| tt-metal        | `3fa4d753550dba1d4aacc9af45b111ae540f63fc` |
| tt-forge-models | `215d1080a28b3d49f8be77cc4243701ca9586b70` |

(`tt-forge-models` is a submodule of `tt-xla`, branch
`arch-c-36-tt-xla-dev/nsmith/hf-bringup-47`)
