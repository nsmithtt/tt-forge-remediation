# Remediation Summary: cross_encoder-passage_ranking-pytorch-xenova_ms-marco-minilm-l-12-v2-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cross_encoder/passage_ranking/pytorch-Xenova/ms-marco-MiniLM-L-12-v2-single_device-inference]

## Result
SILICON_PASS — added diverse sample pairs to fix degenerate single-element PCC=0

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
cross-encoder-single-pair-pcc-undefined

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.95.

## Root cause
The `XENOVA_MS_MARCO_MINILM_L_12_V2` variant's `sample_pairs` contained only one
query-passage pair. With a single pair the model outputs a (1, 1) logit tensor;
PCC of a 1-element tensor is undefined and evaluates to 0.0. The `pretrained_model_name`
for this variant was already pointing to `cross-encoder/ms-marco-MiniLM-L-12-v2`
(the PyTorch source model) in the `hf-bringup-45` base branch, so the model loaded
successfully but the output could not be evaluated. (`Xenova/ms-marco-MiniLM-L-12-v2`
is an ONNX-only HuggingFace repo with no `pytorch_model.bin` or `model.safetensors`.)

## Fix
Added three more diverse query-passage pairs to `sample_pairs` in
`cross_encoder/passage_ranking/pytorch/loader.py` so that the model outputs a
4-element logit tensor, making PCC well-defined and computable.

File changed: `cross_encoder/passage_ranking/pytorch/loader.py`
Repo: `tt-forge-models`
Branch: `remediation/cross_encoder-passage_ranking-pytorch-xenova_ms-marco-minilm-l-12-v2-single_device-inference`
Commit: `43cdd3124351baa046c4b27a3933eceaa2a6cee3`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    73.24s
- Tier A attempts: N/A

## Files changed
- `cross_encoder/passage_ranking/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 43cdd3124351baa046c4b27a3933eceaa2a6cee3 |
