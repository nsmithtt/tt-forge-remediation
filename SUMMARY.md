# Remediation Summary: importlib._bootstrap

## Skill version
6

## Test
importlib._bootstrap

## Result
FAIL — `importlib._bootstrap` is not a real pytest path; it is a bogus test name injected by a CI bringup result-collection bug and can never pass

## Stack layer
loader

## Tier
B

## Bug fingerprint
bogus-test-name-from-bringup-collection

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ERROR: file or directory not found: importlib._bootstrap

## Root cause
`importlib._bootstrap` is Python's internal import bootstrapping module, not a test file. It appears as the warning source in pytest's warnings summary when SWIG types emit `DeprecationWarning` at import time:

```
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
```

The CI bringup pipeline (`setup_cpu_remediator.sh`) generates `collect_vulcan_models_torch.log` by running `pytest -q --collect-only tests/runner/test_models.py::test_all_models_torch ...`. That command's output contains both real test IDs and the pytest warnings-summary block. The downstream `filter_known_tests.py` script reads the log line-by-line and emits any line not already in `results_main.yaml` **without validating that the line is a valid pytest test ID**. As a result, warning-source strings such as `importlib._bootstrap` (or the fuller `<frozen importlib._bootstrap>:488: DeprecationWarning: ...`) leak into the candidate pool, are randomly selected by `pick_one.sh`, and are handed to the Claude model-remediation agent as if they were real model tests.

The agent then calls `pytest importlib._bootstrap -svv`, which immediately fails with `ERROR: file or directory not found: importlib._bootstrap`, logs `MAX_FAILED_ATTEMPTS` or a PASS (if the agent mis-read the exit code), and the entry propagates into `results_main.yaml`. The silicon validator subsequently tries to re-validate the `PASS` entry on silicon, producing the `SILICON_FAIL` record that triggered this remediation.

No change to tt-xla, tt-mlir, tt-metal, or tt-forge-models can make `pytest importlib._bootstrap -svv` succeed, because the path does not exist. The defect lives entirely in the CI bringup tooling.

## Fix
The fix belongs in the CI bringup pipeline, outside all defined compiler-stack layers:

1. **`filter_known_tests.py`** — add a format guard so only lines matching the expected test-ID pattern are emitted as candidates:
   ```python
   TEST_ID_RE = re.compile(r'^tests/')
   for line in sys.stdin:
       stripped = line.rstrip("\n")
       if TEST_ID_RE.match(stripped) and stripped not in known:
           print(stripped)
   ```
2. **`results_main.yaml`** — the existing `importlib._bootstrap` (and similar `<frozen importlib._bootstrap>...`) entries should be deleted or blacklisted so the remediation loop terminates.

## Tier B justification
cross-repo — the defect is in the CI bringup tooling (`.hf-bringup/filter_known_tests.py`), which is outside all defined compiler-stack layers (loader / tt-xla / tt-mlir / tt-metal). There is no change to any of those layers that would make `pytest importlib._bootstrap` pass, because the test path is non-existent.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    0.00s (instant — pytest error before collection)
- Tier A attempts: N/A

## Files changed
None — no fix is possible within the compiler stack.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
