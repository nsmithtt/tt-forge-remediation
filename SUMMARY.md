# Remediation Summary: bartowski_c4ai_command_r_plus_gguf-causal_lm-pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — C4AI Command R+ (~104B params, ~58 GB Q4_K_M) exceeds n150 12 GB DRAM; hardware capacity ceiling

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-command-r-arch-missing-and-hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

(CI failure, 2026-04-23; locally reproduces as OSError because the single-file GGUF was restructured into 6 shards on HuggingFace since then)

## Root cause

Two issues, both in the loader layer:

1. **Missing GGUF architecture registration**: The GGUF file stores `general.architecture = "command-r"`, but
   transformers 5.x `GGUF_SUPPORTED_ARCHITECTURES` (derived from `GGUF_CONFIG_MAPPING`) does not include
   `"command-r"`. The check at `modeling_gguf_pytorch_utils.py:477` raises `ValueError`. Additionally,
   `"command-r"` is the GGUF name for the HuggingFace `"cohere"` model_type; the two names differ and
   `AutoConfig` would fail to find `CohereConfig` unless the output is translated.

2. **GGUF file restructured into shards**: The original single-file `c4ai-command-r-plus-Q4_K_M.gguf` was
   split into 6 shards on HuggingFace. The loader's `GGUF_FILE` pointed to the (now-deleted) single file.

After fixing both loader issues, **the model cannot run on n150**: C4AI Command R+ has ~104B parameters;
at Q4_K_M quantization (~4.5 bits/param) the weights occupy ~58 GB, far exceeding n150's 12 GB DRAM.
This is a hardware capacity ceiling, not a compiler bug.

## Fix

In `tt_forge_models` (remediation branch `remediation/bartowski_c4ai_command_r_plus_gguf-...`):

`bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch/loader.py`:
- Added `_patch_transformers_command_r_gguf()` which:
  - Appends `"command-r"` to `GGUF_SUPPORTED_ARCHITECTURES`
  - Adds the `"command-r"` config field mapping to `GGUF_TO_TRANSFORMERS_MAPPING["config"]`
    (`context_length` → `max_position_embeddings`, `block_count` → `num_hidden_layers`, etc.)
  - Registers `GGUFLlamaConverter` for the Command R SentencePiece tokenizer
  - Wraps `load_gguf_checkpoint` in both `modeling_gguf_pytorch_utils` and
    `configuration_utils` to translate `model_type "command-r"` → `"cohere"` in the
    returned config dict (so `AutoConfig` finds `CohereConfig`)
- Updated `GGUF_FILE` from `"c4ai-command-r-plus-Q4_K_M.gguf"` (single file, deleted) to
  `"c4ai-command-r-plus-Q4_K_M.gguf/c4ai-command-r-plus-Q4_K_M-00001-of-00006.gguf"` (first shard)

In `tt_xla` test config:
- Added `bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch-C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF-single_device-inference`
  as `KNOWN_FAILURE_XFAIL` with reason: model too large for single device.

## Verification
- pytest exit: FAIL (xfailed — test fails as expected with OSError file-not-found for shard download; XFAIL exits 0)
- Hardware: not-run (no silicon run needed for XFAIL disposition; model cannot fit on any single n150/n300 device)
- Duration: 121.79s (xfail run locally)
- Tier A attempts: N/A

## Files changed
- `bartowski_c4ai_command_r_plus_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 032d7fcd5276120672fc1789e5a82ca95533c22d |
| tt-forge-models | 1917b9fda9c1d1e7668bbc432d3182c3c2dd08fe |
