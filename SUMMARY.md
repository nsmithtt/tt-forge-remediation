# Remediation Summary: coca-pytorch-ViT_L_14_laion2B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[coca/pytorch-ViT_L_14_laion2B-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed: hf-hub URL 404, spacy namespace collision, image/text batch-size mismatch in CoCa cross-attention

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
coca-open-clip-hf-hub-url-404

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
FileNotFoundError: Failed to download file (open_clip_config.json) for
laion/CoCa-ViT-L-14-laion2B-s13B-b90k. Last error: 404 Client Error.
Entry Not Found for url: https://huggingface.co/laion/CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/main/open_clip_config.json.
```
(Masked at collection by `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`)

## Root cause
Three independent loader bugs:

1. **Broken hf-hub URL (coca loader)**: `create_model_from_pretrained("hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k")` requires `open_clip_config.json` in the HuggingFace repo, but that file is absent — HTTP 404. The open_clip built-in pretrained registry (`coca_ViT-L-14` / `laion2b_s13b_b90k`) downloads from OpenAI's servers and works correctly.

2. **spacy namespace collision (huspacy loader)**: `huspacy/pytorch/loader.py` had `import spacy` at module level. During pytest collection `dynamic_loader` imports this file, which causes `tt_forge_models/spacy/` to shadow the real `spacy` package via namespace-package mechanics. Any subsequent call to `load_dataset()` triggers `datasets._dill` which checks `issubclass(obj_type, spacy.Language)` and crashes with `AttributeError: module 'spacy' has no attribute 'Language'`.

3. **Image/text batch-size mismatch in CoCa cross-attention**: `load_inputs` returned `pixel_values` with batch_size=1 and `text_tokens` with batch_size=2 (two text prompts). CoCa's `forward(image, text)` runs paired cross-attention — text tokens cross-attend to image tokens — requiring matching batch sizes. With batch=1 image and batch=2 text, `F.multi_head_attention_forward` tries `k.view(255, bsz*num_heads=24, 64)` but the projected key has only 195,840 = 255×768 elements (consistent with batch=1), causing `RuntimeError: shape '[255, 24, 64]' is invalid for input of size 195840`.

## Fix
All fixes are in `tt-xla/third_party/tt_forge_models` on branch `remediation/coca-pytorch-ViT_L_14_laion2B-single_device-inference`:

1. `coca/pytorch/loader.py`: Changed `create_model_from_pretrained("hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k")` to `create_model_from_pretrained("coca_ViT-L-14", pretrained="laion2b_s13b_b90k")` in both `load_model` and `load_inputs`.

2. `huspacy/pytorch/loader.py`: Moved `import spacy` from module level into `_load_nlp()` to prevent namespace collision during collection.

3. `coca/pytorch/loader.py`: Added `pixel_values = pixel_values.repeat(len(self.text_prompts), 1, 1, 1)` in `load_inputs` to replicate the image to match the number of text prompts, ensuring matched batch sizes in CoCa's cross-attention.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    216.45s (0:03:36)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: coca/pytorch/loader.py` (3 commits: hf-hub URL fix, batch-size fix)
- `tt-forge-models: huspacy/pytorch/loader.py` (lazy spacy import)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7b7b4b55665ccf836226f70f874e253fe57698f2 |
| tt-forge-models | e77c2871c6a494bd01c470366c4eade4557e6ac5 |
