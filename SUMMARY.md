# Remediation Summary: forgotten_safeword_gguf-causal_lm-pytorch-70B_V5_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[forgotten_safeword_gguf/causal_lm/pytorch-70B_V5_GGUF-single_device-inference]

## Result
XFAIL — 70B model dequantizes to ~140 GB BF16 at load time, exceeding single-device DRAM on all TT hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-70b-model-exceeds-single-device-dram

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
Two issues:

1. **Loader bug (primary surface failure):** The `forgotten_safeword_gguf/causal_lm/pytorch/` directory had no `requirements.txt`. In a pytest session, `RequirementsManager` installs per-model requirements for each test and rolls them back on exit. When a prior test installed `gguf` (via its own `requirements.txt`) and then rolled it back, the `forgotten_safeword_gguf` test ran in an environment without `gguf`. Transformers' `load_gguf_checkpoint` checks `is_gguf_available()` at entry and raises the ImportError when the package is absent.

2. **Hardware-class capacity ceiling (primary blocker):** The model (`mradermacher/Forgotten-Safeword-70B-v5.0-heretic-GGUF`, Q4_K_M quantization) has 70B parameters. Transformers dequantizes all GGUF weights to BF16 at load time: 70B × 2 bytes ≈ 140 GB. This far exceeds single-device DRAM on all TT hardware (n150: 12 GB, p150b: 24 GB). Even the I1-quantized variant shares the same parameter count and BF16 footprint after dequantization.

## Fix
1. **Loader fix** — Added `forgotten_safeword_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0` in `tt-forge-models` on branch `remediation/forgotten_safeword_gguf-causal_lm-pytorch-70B_V5_GGUF-single_device-inference`.

2. **Test config** — Marked both 70B variants `KNOWN_FAILURE_XFAIL` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` in `tt-xla`:
   - `forgotten_safeword_gguf/causal_lm/pytorch-70B_V5_GGUF-single_device-inference`
   - `forgotten_safeword_gguf/causal_lm/pytorch-70B_V5_I1_GGUF-single_device-inference`

## Verification
- pytest exit: FAIL (not run on silicon — hardware-class XFAIL, model exceeds DRAM)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `forgotten_safeword_gguf/causal_lm/pytorch/requirements.txt` (new file, in tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 914c920b101fef54cdd5d742b799c7882ecb6564 |
| tt-forge-models | 9806ab3d1e2f45571f4fae53cbcca1c19e40660f |
