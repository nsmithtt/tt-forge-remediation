# Remediation Summary: birds_classifier_efficientnet_b2/pytorch-Default-single_device-inference

## Skill version
9

## Test
tests/runner/test_models.py::test_all_models_torch[birds_classifier_efficientnet_b2/pytorch-Default-single_device-inference]

## Result
SILICON_PASS

## Failure
2026-04-25 01:42:07.863 | WARNING  | tests.runner.utils.dynamic_loader:get_model_variants:289 - Cannot import path: /home/ttuser/hf-bringup/tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_nsfw_captioning_v2_i1_gguf/causal_lm/pytorch/loader.py: dictionary changed size during iteration

(Symptom on the configured branch: `AttributeError: module 'spacy' has no attribute 'Language'` in `datasets._dill`, then after that fix: `RuntimeError: shape '[1, 1408]' is invalid for input of size 0` from `AvgPool2d`.)

## Root cause
Two loader-layer bugs, both in `tt_forge_models`:

1. **Namespace package shadowing (huspacy)**: `tt_forge_models/huspacy/pytorch/loader.py` had a top-level `import spacy`. Because `models_root` (the `tt_forge_models` directory) is prepended to `sys.path` during test collection, Python resolves `import spacy` to the `tt_forge_models/spacy/` directory as a namespace package instead of the real `spacy` package. This puts a stub `spacy` namespace package (with no `Language` attribute) in `sys.modules["spacy"]`. When the `birds_classifier` test later calls `load_dataset("huggingface/cats-image")`, the `datasets._dill` serializer checks `if "spacy" in sys.modules` → True, then tries `spacy.Language` → `AttributeError`.

2. **AvgPool2d with oversized kernel (birds_classifier)**: `transformers/models/efficientnet/modeling_efficientnet.py` instantiates `nn.AvgPool2d(config.hidden_dim=1408, ceil_mode=True)` as the global pooler. The EfficientNet-B2 feature map at 224×224 input is 9×9, far smaller than 1408. XLA's `AvgPool2d` implementation returns an empty tensor when `kernel_size > input_size` with `ceil_mode=True`, causing the downstream reshape to `[1, 1408]` to fail with size 0.

## Fix
Both fixes are in `tt_forge_models`, `remediation/birds_classifier_efficientnet_b2-pytorch-Default-single_device-inference`:

1. **`huspacy/pytorch/loader.py`**: Moved top-level `import spacy` inside `_load_nlp()` (deferred import). This prevents the `tt_forge_models/spacy/` namespace package from entering `sys.modules` during collection. Not a forbidden workaround — the loader's functionality is unchanged; it still calls `spacy.load()` at runtime.

2. **`birds_classifier_efficientnet_b2/pytorch/loader.py`**: After loading the model, replaced `model.efficientnet.pooler` (which is `AvgPool2d(1408, ceil_mode=True)`) with `nn.AdaptiveAvgPool2d(1)`. Both are global average pooling over the spatial dimensions and are numerically equivalent. `AdaptiveAvgPool2d` uses a different ATEN op that XLA handles correctly. This is not a forbidden workaround — the model depth, width, inputs, and numeric thresholds are unchanged.

## Verification
pytest exit status: PASSED  
Wall-clock duration: 198.37s (3:18)  
Hardware: n150 (wormhole_b0)

## Files changed
- `tt_forge_models/huspacy/pytorch/loader.py` — defer `import spacy` to `_load_nlp()`
- `tt_forge_models/birds_classifier_efficientnet_b2/pytorch/loader.py` — replace `AvgPool2d(1408)` with `AdaptiveAvgPool2d(1)`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 79e1ef15f6c167155752aaea17c2fbb3ecdc08be |
