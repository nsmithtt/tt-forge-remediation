# Remediation Summary: beetlelm-causal_lm-pytorch-beetlelm_deu_L1_eng_L2_balanced-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[beetlelm/causal_lm/pytorch-beetlelm_deu_L1_eng_L2_balanced-single_device-inference]

## Result
SILICON_PASS — loader bugs fixed + SDPA 2D→4D mask reshape in composite lowering path

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
sdpa-composite-attn-mask-not-4d

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
loc("custom-call.737"): error: 'ttir.scaled_dot_product_attention' op Attention mask must be a 4D tensor
ValueError: Error code: 13
```
(Original reported failure `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a pytest footer warning, not the real error.)

## Root cause

Three stacked bugs:

1. **Loader — missing tokenizer** (`loader` layer): BeetleLM HuggingFace repos contain no tokenizer files. `AutoTokenizer.from_pretrained` on the model repo produced a generic fast-tokenizer stub that couldn't construct a vocabulary backend. Fixed by loading from the paired tokenizer repo (`BeetleLM/bpe_babylm-eng-babylm-deu`) and overriding `pad_token` to `<PAD>` (ID=1, within the model's vocab_size=32000).

2. **Loader — no pretrained weights** (`loader` layer): BeetleLM repos have no `pytorch_model.bin` or `model.safetensors`. `AutoModelForCausalLM.from_pretrained` raised `OSError`. Fixed by using `get_class_from_dynamic_module("pico_decoder.PicoDecoderHF", ...)` to instantiate the class directly with random weights, which is valid for compiler architecture testing.

3. **Compiler — SDPA attention mask rank** (`tt-mlir` layer): The `PicoDecoder` builds a 2D `(S, S)` causal mask and passes it to `F.scaled_dot_product_attention`. The XLA bridge emits this as a `stablehlo.composite` op named `tenstorrent.scaled_dot_product_attention`. The composite lowering pass (`TenstorrentScaledDotProductAttentionConversionPattern` in `StableHLOLegalizeCompositePass.cpp`) forwarded the 2D mask directly to `ttir::ScaledDotProductAttentionOp`, which requires 4D `(B, H, S, S)`. The TTIR verifier rejected it. Fixed by inserting a `ttir::ReshapeOp` that prepends `(4 - rank)` broadcast dimensions of size 1 before the TTIR op is created. The same reshape was also added to the `StableHLOToTTIRScaledDotProductAttentionOpConversionPattern` (custom-call path) for completeness.

## Fix

**Loader fix** (`tt-forge-models`, branch `remediation/beetlelm-causal_lm-pytorch-beetlelm_deu_L1_eng_L2_balanced-single_device-inference`, commit `ff76ba7f22`):
- `beetlelm/causal_lm/pytorch/loader.py`: load tokenizer from paired `bpe_babylm-*` repos; instantiate model via `get_class_from_dynamic_module` with random weights.

**Compiler fix** (`tt-mlir`, branch `remediation/beetlelm-causal_lm-pytorch-beetlelm_deu_L1_eng_L2_balanced-single_device-inference`, commit `15b20fd70`):
- `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`: add `ttir::ReshapeOp` to expand <4D attention masks to 4D in `TenstorrentScaledDotProductAttentionConversionPattern`.
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`: same reshape in `StableHLOToTTIRScaledDotProductAttentionOpConversionPattern` (defensive, custom-call path).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    59.95s
- Tier A attempts: 1

## Files changed
- `tt-forge-models`: `beetlelm/causal_lm/pytorch/loader.py`
- `tt-mlir`: `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`
- `tt-mlir`: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 15b20fd7037c09e51af3106f451d2ff4b020a22a |
| tt-xla          | 6b86ca80cddc24992bcbc6688e2aef8926a891bf |
| tt-forge-models | ff76ba7f220b6326694db3aa4708d0e8686d0bd1 |
