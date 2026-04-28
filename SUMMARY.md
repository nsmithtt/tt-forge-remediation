# Remediation Summary: frozen-importlib-swigpypacked-deprecation-warning

## Skill version
6

## Test
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

## Result
NO_FIX_NEEDED — not a real pytest test; this is a Python import-time warning captured as a test name by the bringup runner

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
/bin/bash: line 1: frozen: No such file or directory

## Root cause
The string `<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute` is a Python import-time deprecation warning emitted when a SWIG-based extension module (`SwigPyPacked`) is imported without a `__module__` attribute. This warning line was incorrectly captured as a pytest test name by the HF bringup runner's output parser.

When the runner subsequently attempted to execute `pytest -svv <frozen importlib._bootstrap>:488: ...`, bash tried to treat `frozen` (the first token after `<`) as a shell command, producing `/bin/bash: line 1: frozen: No such file or directory`. There is no actual test with this name, and no model or compiler bug to fix.

The bug is in the bringup runner's output-parsing logic, which is capturing lines from pytest's warning summary block as test IDs. That output appears at the bottom of pytest's output in a section beginning with `warnings summary`, just before the final summary line. The runner should skip any captured "test name" that does not begin with `tests/`.

## Fix
No fix needed in the compiler stack or model loaders. The upstream bringup runner should filter captured test names to those matching the `tests/` prefix (or the pytest node-id pattern `<path>::<test>`).

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: n/a
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
