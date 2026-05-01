# Remediation Summary: impresso_langident-language_identification-pytorch-impresso-project-language-identifier-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[impresso_langident/language_identification/pytorch-impresso-project/language-identifier-single_device-inference]

## Result
FAIL — model is a floret/fasttext SWIG C extension with string I/O; cannot be compiled or compared as tensors

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-non-tensor-io-swig-model

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

Full error:
TypeError: equal(): argument 'input' (position 1) must be Tensor, not str

Traceback (excerpt):
  tests/infra/evaluators/torch_comparison_evaluator.py:93: in _equal_leaf
      return torch.equal(x, y)
  TypeError: equal(): argument 'input' (position 1) must be Tensor, not str

## Root cause
The `impresso-project/language-identifier` model is built on floret/fasttext — a SWIG-wrapped C
extension — not a differentiable PyTorch computation graph. Its custom `LangDetectorModel`
(loaded via `trust_remote_code=True`) calls `floret.load_model()` in `__init__` and
`floret.predict(texts, k=1)` in `forward()`. Consequently:

1. `load_inputs()` returns `{"input_ids": <raw_string>}` — not a tensor.
2. `forward(input_ids)` only accepts Python strings; it cannot operate on XLA symbolic
   tensors, so torch.compile falls back to CPU-eager execution rather than compiling
   anything to TT silicon.
3. `forward()` returns `(list_of_strings, list_of_floats)` — not PyTorch tensors — so
   `TorchComparisonEvaluator._compare_equal` calls `torch.equal(str, str)` and raises
   `TypeError`.

The "swigvarlink" DeprecationWarning that appears as the reported failure message is emitted
by Python 3.12 at session end when the SWIG-internal `swigvarlink` type (part of the floret
package) is garbage-collected. It is not itself the error, but is the last line of pytest's
output, which is what the CI reports as the failure message.

## Fix
No legitimate fix exists in the loader layer: the model's computation lives entirely inside a
SWIG C extension (`floret.predict`), its I/O is strings and floats, and no part of it passes
through the PyTorch graph. There is no path to TT silicon compilation.

The model should be removed from the tt-forge-models test suite, or replaced with a
PyTorch-native language-identification model (e.g., XLM-RoBERTa fine-tuned for language
detection) that produces tensor I/O. Either change is a design decision for the test-suite
owners, not a compiler-stack fix.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    26.38s
- Tier A attempts: N/A

## Files changed
None — no fix was applied.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 706546ab8e0f952decd92e49381988ee19db367f |
