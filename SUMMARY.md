# Remediation Summary: babyhercules_8x150m_gguf-causal_lm-pytorch-8x150M_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[babyhercules_8x150m_gguf/causal_lm/pytorch-8x150M_Q4_K_M-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the current stack; gguf>=0.10.0 was added to tt-xla dev requirements (cd8104788) one day after the failing run

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
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
The failure occurred on 2026-04-22 when the tt-xla venv did not include `gguf>=0.10.0` in its dev requirements. `AutoModelForCausalLM.from_pretrained` with `gguf_file=` raises ImportError if the `gguf` package is absent. Commit `cd8104788` ("Add gguf>=0.10.0 to dev requirements for diffusers GGUF model support") was merged to tt-xla on 2026-04-23, one day after the hf-bringup-29 run recorded the failure. The current stack has gguf 0.18.0 installed and the test passes cleanly.

## Fix
No fix required. The fix is already present in tt-xla at commit `cd8104788` which added `gguf>=0.10.0` to `venv/requirements-dev.txt`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    76.22s (0:01:16)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c9b45c4dfe71bf9beed21e9db576f2728db20aeb |
