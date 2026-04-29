# Remediation Summary: bge_m3_gguf-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bge_m3_gguf/pytorch-Q4_K_M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-bert-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: GGUF model with architecture bert is not supported yet.

If that were fixed, the next failure would be:
NotImplementedError: Unknown gguf model_type: bert in gguf-py. (raised in get_gguf_hf_weights_map when the outermost GGUF patcher chain link rejected model_to_load kwarg with TypeError)

## Root cause
Two loader bugs:

1. **bert GGUF architecture not registered** (primary): `transformers.modeling_gguf_pytorch_utils.GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_CONFIG_MAPPING` have no `bert` entry. BGE-M3's GGUF file sets `general.architecture = bert`. `AutoTokenizer.from_pretrained(..., gguf_file=...)` calls `load_gguf_checkpoint` which raised `ValueError: GGUF model with architecture bert is not supported yet.`

2. **GGUF patcher chain missing `**kwargs`** (secondary): 26 loaders in tt_forge_models patched `load_gguf_checkpoint` at module-import time with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` — a fixed signature that drops `model_to_load`. When transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` for the actual weight-loading pass, the outermost patcher in the collection chain raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

Additional loader issues already fixed in the existing branch commit:
- Tokenizer loaded from GGUF was unsupported (`bert` arch + SentencePiece mismatch); fixed by loading tokenizer from `BAAI/bge-m3` instead.
- `token_types.weight` stored as 1-D `[1024]` in GGUF but `BertModel` expects `[type_vocab_size, hidden_size]`; fixed with a custom `_BertTensorProcessor` that reshapes it and `GGUF_CONFIG_DEFAULTS_MAPPING["bert"] = {"type_vocab_size": 1}`.
- GGUF file has no pooler weights; `add_pooling_layer=False` prevents randomly-initialized pooler from poisoning the PCC comparison.

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch `remediation/bge_m3_gguf-pytorch-Q4_K_M-single_device-inference`.

**Commit 1** (`5bf85bdd68`) — `bge_m3_gguf/pytorch/loader.py`:
- Registered `bert` in `GGUF_CONFIG_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_CONFIG_DEFAULTS_MAPPING`, `GGUF_TO_FAST_CONVERTERS`, and `TENSOR_PROCESSORS`.
- Added `_BertTensorProcessor` to reshape 1-D `token_types.weight` → `[1, hidden_size]`.
- Changed `_load_tokenizer` to load from `BAAI/bge-m3` (has standard HF tokenizer files; GGUF repo has none).

**Commit 2** (`9eef64df72`) — `bge_m3_gguf/pytorch/loader.py` + 26 other GGUF loaders:
- Added `unpack_forward_output` to handle `BaseModelOutputWithPoolingAndCrossAttentions`.
- Added `add_pooling_layer=False` to `from_pretrained` call.
- Applied `**kwargs` fix to 26 GGUF loaders that had the old fixed-signature `_patched_load_gguf_checkpoint` incompatible with transformers 5.x `model_to_load` kwarg.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 227.88s (0:03:47)
- Tier A attempts: N/A

## Files changed
- `bge_m3_gguf/pytorch/loader.py`
- 26 × `*/causal_lm/pytorch/loader.py` (GGUF loaders — **kwargs fix only)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 54011d600edcc6908fe13d3d1dc917524f29da21 |
| tt-forge-models | 9eef64df721cf8023dd6a6c9f69ce7e62ca6274d |
