# Remediation Summary: deepseek/deepseek_ocr_2/pytorch-Ocr2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_ocr_2/pytorch-Ocr2-single_device-inference]

## Result
FAIL — CPU BF16 forward pass produces all-NaN logits; pcc=nan on TT run after loader crop-size fix

## Stack layer
loader

## Tier
B

## Bug fingerprint
bf16-cpu-nan-forward-pass

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `raise ImportError(`
Actual reproduced failure (first): `UnboundLocalError: cannot access local variable 'param_img' where it is not associated with a value`
Actual reproduced failure (second, after loader fix): `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.`

## Root cause

**Bug 1 (fixed):** `deepencoderv2.py`'s `Qwen2Decoder2Encoder.forward` only handles `n_query == 144` (from 768 px crops via patch_size=16 × 2× stride-2 downsampling → 12×12) and `n_query == 256` (from 1024 px global view → 16×16). The loader called `preprocess(image_size=640)`, and `dynamic_preprocess` used its own default of 640 px for crop size (the `image_size` argument was not forwarded). A 640 px crop through `ImageEncoderViT` → `net_2` (stride 2) → `net_3` (stride 2) gives 640/16/2/2 = 10 → n_query = 100, which is neither 144 nor 256, so `param_img` was never assigned → `UnboundLocalError`.

**Bug 2 (unfixed, Tier B):** After fixing the crop size to 768 px (matching `query_768`, n_query=144), the model compiles and runs on TT silicon but the CPU golden reference forward pass itself produces all-NaN logits in BF16. The NaN propagates through all output logits. The model is a large multimodal OCR model (DeepSeek-V2 backbone with MLA attention + custom `deepencoderv2` encoder). The NaN is consistent with BF16 overflow or instability in one of the attention softmax or projection operations; without further diagnostic tooling (e.g., `torch.autograd.set_detect_anomaly(True)` on GPU, or layer-by-layer nan detection), the exact source cannot be pinpointed.

## Fix

**Bug 1 — Applied:**
1. `third_party/tt_forge_models/deepseek/deepseek_ocr/pytorch/src/model_utils.py`: Changed `dynamic_preprocess(image)` to `dynamic_preprocess(image, image_size=image_size)` so the crop size is forwarded from the `preprocess()` caller.
2. `third_party/tt_forge_models/deepseek/deepseek_ocr_2/pytorch/loader.py`: Changed `image_size=640` to `image_size=768` so crops are 768×768, yielding n_query = 768/16/2/2 × 768/16/2/2 = 12×12 = 144, matching `deepencoderv2`'s `query_768 = nn.Embedding(144, hidden_dimension)`.

Branch: `remediation/deepseek_ocr_2-pytorch-Ocr2-single_device-inference` in tt_forge_models at commit `a227df4b83d71999f6c5ce0d343997cb473d10ec`.

**Bug 2 — Proposed fix location:** Identify the specific attention block producing NaN (either the custom `deepencoderv2` Qwen2 decoder using SDPA with `token_type_ids`-based mask, or the main DeepSeek-V2 MLA attention on the multimodal token sequence). Candidate fix: run the reference computation in float32, add logit clamping/softclamp in MLA attention, or verify if the model requires CUDA-level tensor-core BF16 accumulation precision that CPU lacks.

## Tier B justification (FAIL with Tier=B only)
`internal-error-unknown-mechanism`

The NaN originates in the CPU forward pass before TT even runs. The model has 2707 weight tensors and multiple attention stacks (deepencoderv2 Qwen2 decoder + DeepSeek-V2 MLA language model). Pinpointing the exact NaN source requires dedicated diagnosis-first work (per-layer nan checking, GPU-based anomaly detection) that is beyond a single-fix scope.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    918.55s (0:15:18)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/deepseek/deepseek_ocr/pytorch/src/model_utils.py` — forward `image_size` to `dynamic_preprocess`
- `third_party/tt_forge_models/deepseek/deepseek_ocr_2/pytorch/loader.py` — change `image_size=640` to `image_size=768`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | a227df4b83d71999f6c5ce0d343997cb473d10ec (branch: remediation/deepseek_ocr_2-pytorch-Ocr2-single_device-inference) |
