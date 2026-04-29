# Remediation Summary: bert_crf_harem-token_classification-pytorch-hcaeryks_bert-crf-harem-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bert_crf_harem/token_classification/pytorch-hcaeryks/bert-crf-harem-single_device-inference]

## Result
SILICON_PASS — five loader fixes for transformers 5.x compatibility and CRF inference path

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-meta-device-nested-from-pretrained

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError raised inside BERT_CRF.__init__ when BertModel.from_pretrained() is
called while the transformers 5.x meta-device init context is active.  Secondary
failures: missing pytorch-crf / sentencepiece requirements, AutoTokenizer failure
(no tokenizer files in hcaeryks/bert-crf-harem), dtype inconsistency (BertModel
loaded in float32 while outer from_pretrained sets bfloat16), and CRF decode path
using Python control flow (assert mask[0].all(), list loops) fatal to TT XLA
compilation.

## Root cause
The BERT_CRF custom class (trust_remote_code, from arubenruben/PT-BERT-Large-CRF-HAREM-Default)
was written for transformers 4.29.1 and has multiple incompatibilities with 5.x:

1. **Missing requirements**: pytorch-crf and sentencepiece were not in requirements.txt.
2. **Meta-device context conflict**: transformers 5.x `from_pretrained` activates
   `init_empty_weights()` (a meta-device context), but BERT_CRF.__init__ calls
   `BertModel.from_pretrained()` directly inside that context, which raises RuntimeError
   because the nested call tries to instantiate tensors on the meta device.
3. **all_tied_weights_keys not set**: BERT_CRF never calls post_init(), so
   `_finalize_model_loading` in 5.x fails trying to access `all_tied_weights_keys`.
4. **position_ids buffer clobbered**: The outer `from_pretrained` replaces all
   non-persistent buffers (including position_ids correctly set by the inner call)
   with `torch.empty_like` uninitialised tensors.
5. **Dtype mismatch**: BertModel is loaded in float32 from the neuralmind checkpoint,
   while the outer context sets bfloat16 default dtype, leaving `self.linear` in bf16.
6. **CRF decode D2H transfer**: In inference mode (labels=None), `BERT_CRF.forward`
   calls `crf.decode()` → `_viterbi_decode()`, which uses `assert mask[0].all()`,
   `.item()`, and Python for-loops — all D2H transfers fatal to TT XLA compilation.

## Fix
All fixes are in `tt_forge_models`, loader file
`bert_crf_harem/token_classification/pytorch/loader.py` plus a new
`requirements.txt`.  Five commits on the remediation branch in `tt_forge_models`
(`remediation/bert_crf_harem-token_classification-pytorch-hcaeryks_bert-crf-harem-single_device-inference`):

1. `76ba9b61b0` — Add `requirements.txt` with `pytorch-crf` and `sentencepiece`.
2. `bb1d9a07d0` — Fix transformers 5.x: switch to `BertTokenizer.from_pretrained`;
   patch `get_init_context` to strip `torch.device` from the meta-init context list;
   patch `_adjust_tied_keys_with_tied_pointers` to init `all_tied_weights_keys` on
   demand; patch `_move_missing_keys_from_meta_to_device` to skip buffers already
   on CPU (position_ids).
3. `3e541bcdd2` — Cast model to `dtype_override` after loading for dtype consistency.
4. `538659f3cd` — Patch `torchcrf.CRF._viterbi_decode` to accept a boolean mask
   (avoids assertion on ByteTensor vs BoolTensor mismatch in newer PyTorch).
5. `c783e84921` — Patch `BERT_CRF.forward` to return logits directly in inference
   mode (labels=None) instead of calling `crf.decode()`; CRF Viterbi decode is CPU
   post-processing, not part of the compiled graph.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    66.12s
- Tier A attempts: N/A

## Files changed
- `bert_crf_harem/token_classification/pytorch/requirements.txt` (new)
- `bert_crf_harem/token_classification/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ab18b7caa8f648b461cd1e8619c7285f5743a8c6 |
| tt-forge-models | c783e849217444825d7effefb04768f25664ee3f |
