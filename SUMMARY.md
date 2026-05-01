# Remediation Summary: bge_1_5-embedding_generation-pytorch-Qdrant_Small_Zh_v1_5-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bge_1_5/embedding_generation/pytorch-Qdrant_Small_Zh_v1_5-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
qdrant-bge-small-zh-onnx-only-repo

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: Qdrant/bge-small-zh-v1.5 does not appear to have a file named pytorch_model.bin or model.safetensors.

## Root cause
`Qdrant/bge-small-zh-v1.5` on HuggingFace Hub is an ONNX-optimized export of the original
`BAAI/bge-small-zh-v1.5` model. The repository ships only `model_optimized.onnx` — there are
no PyTorch weights (`pytorch_model.bin` or `model.safetensors`). The loader's `ModelConfig` for
the `QDRANT_BGE_SMALL_ZH_V1_5` variant incorrectly used `Qdrant/bge-small-zh-v1.5` as the
`pretrained_model_name`, causing `AutoModel.from_pretrained` to fail. The Qdrant repo's own
`config.json._name_or_path` is `BAAI/bge-small-zh-v1.5`, confirming that the BAAI repo is the
correct PyTorch weight source.

## Fix
Changed `pretrained_model_name` for `ModelVariant.QDRANT_BGE_SMALL_ZH_V1_5` from
`"Qdrant/bge-small-zh-v1.5"` to `"BAAI/bge-small-zh-v1.5"` in:

`tt_forge_models/bge_1_5/embedding_generation/pytorch/loader.py`

Branch: `remediation/bge_1_5-embedding_generation-pytorch-Qdrant_Small_Zh_v1_5-single_device-inference`
in the `tenstorrent/tt-forge-models` repository.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    45.65s
- Tier A attempts: N/A

## Files changed
- tt_forge_models/bge_1_5/embedding_generation/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e8d41c7f0033cc1c4888d9cbab91f27d6cf50ae0 |
| tt-forge-models | b28ee657f6445e900a9cf736adbd8f3cffac6c18 |
