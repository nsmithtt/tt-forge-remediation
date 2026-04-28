# Remediation Summary: beetlelm-causal_lm-pytorch-beetlelm_nld-bul_balanced-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beetlelm/causal_lm/pytorch-beetlelm_nld-bul_balanced-single_device-inference]

## Result
FAIL — model weights absent from HuggingFace Hub; tokenizer source fix committed but model cannot be loaded for hardware testing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
beetlelm-missing-model-weights-on-hub

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure: E   ValueError: Error code: 13

Reproduced as: OSError: BeetleLM/beetlelm_nld-bul_balanced does not appear to have a file named pytorch_model.bin or model.safetensors.

(After tokenizer fix; before fix: ValueError: Couldn't instantiate the backend tokenizer from one of: (1) a `tokenizers` library serialization file, (2) a slow tokenizer instance to convert or (3) an equivalent slow tokenizer class to instantiate and convert.)

## Root cause
Two loader-layer bugs were found:

1. **Wrong tokenizer source (fixed)**: The loader calls `AutoTokenizer.from_pretrained` with the model repo name (`BeetleLM/beetlelm_nld-bul_balanced`), but this repo ships no tokenizer files. The BeetleLM organization stores tokenizers in separate BPE repos (e.g. `BeetleLM/bpe_babylm-nld-babylm-bul`). All BeetleLM model repos confirmed to have no tokenizer files via `list_repo_files`.

2. **Model weights absent from HuggingFace Hub (unfixable in loader)**: `BeetleLM/beetlelm_nld-bul_balanced` contains only `config.json` and `pico_decoder.py` — no `.safetensors` or `.bin` weights file. All inspected BeetleLM model variants (`beetlelm_nld-bul_balanced`, `beetlelm_nld-bul_heritage`, `beetlelm_eng-bul_balanced`, `beetlelm_bul-deu_balanced`, `beetlelm_deu_L1-eng_L2_balanced`) are missing weights. The models were registered on HuggingFace Hub with architecture code only; trained weights were never uploaded.

The original CI error `ValueError: Error code: 13` is a PJRT XLA compilation error (fired from `_xla_warm_up_cache`, `_xla_step_marker`, or `_xla_sync_multi`). It indicates the model loaded in CI (from a previously cached state) but failed during StableHLO compilation on TT hardware. That compiler-stack failure cannot be diagnosed or triaged without loadable model artifacts.

## Fix
**Applied**: Updated `beetlelm/causal_lm/pytorch/loader.py` in tt_forge_models to add a `_TOKENIZER_NAMES` dict mapping each variant to its correct BPE tokenizer repo, and updated `_load_tokenizer` to use that dict instead of the model repo name. Commit `4fcdede43d` in tt_forge_models, branch `remediation/beetlelm-causal_lm-pytorch-beetlelm_nld-bul_balanced-single_device-inference`.

**Remaining (not fixable in loader)**: Model weights for `BeetleLM/beetlelm_nld-bul_balanced` are not present on HuggingFace Hub. The fix would require the BeetleLM authors to upload trained weights to the HuggingFace repository. There is no alternative public source for these weights.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    13.34s (loader failure before hardware)
- Tier A attempts: N/A

## Files changed
- `beetlelm/causal_lm/pytorch/loader.py` (tt_forge_models): add `_TOKENIZER_NAMES` dict and use it in `_load_tokenizer`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d52e2fa614d150431ee803447409e3c580199e01 |
| tt-forge-models | 4fcdede43df62ea24256eb644a61ec60f67a6f34 |
