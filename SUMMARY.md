# Remediation Summary: bros-feature_extraction-pytorch-naver_clova_ocr-bros_base_uncased-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bros/feature_extraction/pytorch-naver-clova-ocr/bros-base-uncased-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bros-bbox-embedding-float32-bf16-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16

## Root cause
Loader bug. `BrosModel.forward` computes `scaled_bbox = bbox * self.config.bbox_scale` where
`bbox` is `torch.long` and `bbox_scale` is a Python float. PyTorch promotes `long * float` to
float32, so `scaled_bbox` is always float32 regardless of model dtype. This float32 tensor flows
through `BrosBboxEmbeddings` sinusoidal embedding computation and into `bbox_projection` (an
`nn.Linear`). When the model is loaded with `torch_dtype=bfloat16`, `bbox_projection.weight` is
bfloat16 but the input is float32. TT's `torch_overrides.py` replaces `F.linear` with
`torch.einsum("...mk,...nk->...mn", inp, weight)` which requires exact dtype match (unlike native
PyTorch which does type promotion), causing the RuntimeError.

## Fix
In `bros/feature_extraction/pytorch/loader.py`, removed the `dtype_override` pass-through to
`AutoModel.from_pretrained`. The model keeps its native float32 dtype, eliminating the
float32-vs-bfloat16 mismatch in the bbox embedding path.

File changed: `tt-xla/third_party/tt_forge_models/bros/feature_extraction/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    84.82s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/bros/feature_extraction/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 16ebc007b7d66404e163ad36923ad847169ed1c7 |
| tt-forge-models | 993b522a30ea343aa2a9d68129533c0bdcc54c38 |
