# Remediation Summary: pi_05-pytorch-rayhanfahmed_pi05_flow_v2_feb24-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[pi_05/pytorch-rayhanfahmed_pi05_flow_v2_feb24-single_device-inference]

## Result
FAIL — PaliGemma tokenizer (google/paligemma-3b-pt-224) is gated; two loader bugs fixed but full silicon run requires CI HF_TOKEN for gated model access

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
spacy-namespace-shadows-real-package, lerobot-policy-preprocessor-json-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AttributeError: module 'spacy' has no attribute 'Language'

## Root cause
Two stacked loader bugs:

**Bug 1 (reported failure — spacy namespace shadowing):** `DynamicLoader.setup_models_path` in `tests/runner/utils/dynamic_loader.py` inserts `models_root` into `sys.path`. Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python registers it as a namespace package named `spacy`. When `huspacy/pytorch/loader.py` does `import spacy` at module-level during test collection, this namespace package is stored in `sys.modules['spacy']` without `Language`. Later, when `LeRobotDataset("lerobot/libero")` is loaded in the pi_05 test, `datasets._dill.save()` checks `if "spacy" in sys.modules` and calls `spacy.Language`, triggering AttributeError.

**Bug 2 (underlying lerobot API mismatch):** `rayhanfahmed/pi05-flow-v2-feb24` was saved with an older lerobot format that stores normalisation statistics in `norm_stats.json`. Current lerobot (≥0.4.0) `make_pre_post_processors(config, pretrained_model_name, ...)` calls `PolicyProcessorPipeline.from_pretrained` which looks for `policy_preprocessor.json` on HuggingFace Hub. That file does not exist in the checkpoint, causing `FileNotFoundError`. This error is masked in CI by Bug 1 (spacy error fires first), but is the next failure after Bug 1 is fixed.

## Fix
**Fix 1 — `tt-xla/tests/runner/utils/dynamic_loader.py`:** Remove `sys.path.insert(0, models_root)` from `setup_models_path`. Relative imports in loaders work via `__package__` and the manually-registered `tt_forge_models` namespace package; the `sys.path` insertion is not needed and causes subdirectory names to shadow installed packages.

**Fix 2 — `tt_forge_models/pi_05/pytorch/loader.py`:** In `load_inputs`, instead of passing `pretrained_model_name` to `make_pre_post_processors` (which triggers the `policy_preprocessor.json` lookup), download `norm_stats.json` from the checkpoint repo via `hf_hub_download`, convert the quantile stat lists to `torch.Tensor`, and pass them as `dataset_stats=`. This lets lerobot build the preprocessor pipeline from the config directly, bypassing the missing `policy_preprocessor.json`.

Remediation branch: `remediation/pi_05-pytorch-rayhanfahmed_pi05_flow_v2_feb24-single_device-inference` in both tt-xla and tt-forge-models.

Local reproduction of the full run was blocked by a third issue: `TokenizerProcessorStep(tokenizer_name="google/paligemma-3b-pt-224")` inside `make_pi05_pre_post_processors` tries to download the gated `google/paligemma-3b-pt-224` tokenizer. This requires `HF_TOKEN` with gated model access, which is available in CI but not locally. This is an environment limitation, not a new bug.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: n/a (blocked by gated model access locally)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py` — remove sys.path.insert(0, models_root)
- `tt_forge_models/pi_05/pytorch/loader.py` — load norm_stats.json and pass as dataset_stats

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4ec79439663f81478e5be457dc2d1deb73af2c3e |
| tt-forge-models | 0e601bd650e9d156f2057a2715a9e81cf4dd8527 |
