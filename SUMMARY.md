# Remediation Summary: apertus_8b_instruct_bnb_4bit-causal_lm-pytorch-Apertus-8B-Instruct-2509-unsloth-bnb-4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[apertus_8b_instruct_bnb_4bit/causal_lm/pytorch-Apertus-8B-Instruct-2509-unsloth-bnb-4bit-single_device-inference]

## Result
FAIL — WH BF16 matmul precision gives PCC=0.978 vs required 0.99; loader fix complete

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
raise ImportError(
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

After loader fix, remaining failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9781359666162491. Required: pcc=0.99.
```

## Root cause
Two separate bugs:

**Bug 1 (loader — fixed):** The `apertus_8b_instruct_bnb_4bit` loader had no `requirements.txt`,
so `bitsandbytes` was never installed before the test ran. `AutoModelForCausalLM.from_pretrained`
detects the `quantization_config` in the model's HuggingFace config, calls
`validate_environment()` in `quantizer_bnb_4bit.py`, and raises `ImportError` when bitsandbytes
is absent. Additionally, `Params4bit.detach()` returns a plain `Tensor` rather than a `Params4bit`,
which breaks `Parameter.__new__`'s type-consistency check in PyTorch 2.7 when the model is moved
to the TT XLA device. A third issue was that the original fix placed `import bitsandbytes as bnb`
at module top level, preventing test collection before the requirements.txt is processed.

**Bug 2 (tt-mlir — unfixed):** After the loader fixes, the model compiles and runs on TT silicon
but gives PCC=0.978 against the CPU FP32 reference. CPU BF16 vs CPU FP32 PCC for the same model
is 0.997, meaning the BF16 dtype floor alone explains only 0.003 of degradation. The additional
~0.019 gap (0.997 − 0.978) between CPU BF16 and TT silicon BF16 is the Wormhole BF16 matmul
precision issue: the TT compiler lowers matmuls to BF16 intermediate accumulation instead of FP32,
accumulating rounding error across 32 transformer layers of an 8B model.

## Fix
**Loader fix — applied in tt_forge_models (remediation branch):**

1. `apertus_8b_instruct_bnb_4bit/causal_lm/pytorch/requirements.txt` — created with
   `bitsandbytes>=0.46.1` so the package is installed before `load_model()` runs.
2. `apertus_8b_instruct_bnb_4bit/causal_lm/pytorch/loader.py` — added
   `_dequantize_bnb4_to_bf16()` that replaces all `Linear4bit` modules with standard `nn.Linear`
   BF16 layers by calling `bnb.functional.dequantize_4bit()`. Called immediately after
   `from_pretrained()`, before `model.eval()`.
3. Same file — moved `import bitsandbytes as bnb` from module top level into the body of
   `_dequantize_bnb4_to_bf16()` so the loader module can be imported during test collection
   (before requirements are processed) without `ImportError`.

**Compiler fix — not attempted (Tier B):**

The WH BF16 matmul precision issue requires preserving FP32 accumulation through all StableHLO→TTIR
lowering passes. This is a cross-cutting change. The proposed fix would be in tt-mlir's matmul
lowering to use FP32 accumulators, but this affects the entire model compilation pipeline.

## Tier B justification
cross-cutting — Preserving FP32 accumulation requires changes across every matmul lowering pass
in tt-mlir, affecting all models that use BF16 weights on Wormhole hardware. This is the same
`ttmlir-f32-precision-not-preserved` bug already documented for Gemma 7B, Qwen3 4B, GPT-J 6B,
and DNABERT-S.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    155.10s (0:02:35)
- Tier A attempts: N/A

## Files changed
- `apertus_8b_instruct_bnb_4bit/causal_lm/pytorch/requirements.txt` (created, in tt_forge_models)
- `apertus_8b_instruct_bnb_4bit/causal_lm/pytorch/loader.py` (modified, in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f80a108bf912bbd96b86f69d7ad734cc3e1a1366 |
| tt-forge-models | a4eb64c4548afa38f7a6befe6520a9cc1df6521a |
