# Remediation Summary: bartowski_liquidai_lfm2_2_6b_exp_gguf-causal_lm-pytorch-LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_liquidai_lfm2_2_6b_exp_gguf/causal_lm/pytorch-LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-lfm2-tokenizer-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

(preceded by intermediate failures during debugging: KeyError: 'lfm2' and IndexError: index out of range in self)

## Root cause
Three chained loader bugs:

1. **KeyError: 'lfm2'** — The LFM2 GGUF file uses `general.architecture = "lfm2"` and `tokenizer.ggml.model = "gpt2"`, but `transformers` does not register `"lfm2"` in `GGUF_TO_FAST_CONVERTERS`. `AutoTokenizer.from_pretrained(..., gguf_file=...)` tries `GGUF_TO_FAST_CONVERTERS["lfm2"]` and raises `KeyError`.

2. **IndexError: index out of range in self** — `GGUFGPTConverter` adds two special tokens to the vocabulary that are not reflected in the GGUF `vocab_size` field. The tokenizer ends up with 65538 tokens while the model embedding table has only 65536 rows. Token IDs for the added special tokens are out of range.

3. **TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache** — `Lfm2Model.forward()` creates an `Lfm2HybridConvCache` object when `use_cache=True` (the default). This custom cache class is not a registered pytree node, so `torch.utils._pytree.tree_map` treats it as a leaf and calls `torch.equal(cache, cache)`, which raises `TypeError`.

## Fix
All three fixes are in `bartowski_liquidai_lfm2_2_6b_exp_gguf/causal_lm/pytorch/loader.py` in `tt-forge-models`:

1. Register the converter at import time:
   ```python
   from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
   GGUF_TO_FAST_CONVERTERS.setdefault("lfm2", GGUFGPTConverter)
   ```

2. Resize embeddings after `from_pretrained`:
   ```python
   if self.tokenizer is not None and len(self.tokenizer) > model.config.vocab_size:
       model.resize_token_embeddings(len(self.tokenizer))
   ```

3. Add `use_cache=False` to the inputs dict in `load_inputs()` to prevent the model from returning an `Lfm2HybridConvCache` object.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    204.30s (0:03:24)
- Tier A attempts: N/A

## Files changed
- `bartowski_liquidai_lfm2_2_6b_exp_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0bffc9ca65524877af77fdbf79090de123453b10 |
| tt-forge-models | ac304846c6eb469e90026b5edaf569aa314b6a5f |
