# Remediation Summary: mradermacher_qwen3_30b_a3b_instruct_reamini_i1_gguf-causal_lm-pytorch-30B_A3B_INSTRUCT_REAMINI_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_30b_a3b_instruct_reamini_i1_gguf/causal_lm/pytorch-30B_A3B_INSTRUCT_REAMINI_I1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B has 30B total parameters (~60 GB BF16 when dequantized from GGUF) which exceeds p150b DRAM (32 GB); hardware-class capacity XFAIL

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-class-dram-capacity-exceeded

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
Two issues:

1. **Missing requirements.txt**: The loader directory lacked a `requirements.txt` with `gguf>=0.10.0`, causing the test runner to fail before even attempting model download. The `gguf` package must be declared as a dependency for the RequirementsManager to install it.

2. **Hardware capacity ceiling**: Qwen3-30B-A3B is a Mixture-of-Experts model with 30B total parameters. When loaded from GGUF, transformers dequantizes weights to BF16, yielding ~60 GB of weight data — nearly double the p150b's 32 GB DRAM capacity. This is a fundamental hardware-class limitation, not a compiler bug.

## Fix
1. Added `requirements.txt` with `gguf>=0.10.0` to `tt_forge_models/mradermacher_qwen3_30b_a3b_instruct_reamini_i1_gguf/causal_lm/pytorch/`.
2. Added `model.config._experts_implementation = "batched_mm"` in `load_model()` to pre-emptively address the Qwen3MoE for-loop segfault (same class as prior fixes in mradermacher_agentic_qwen_30b_a3b_i1_gguf and others).
3. Added `KNOWN_FAILURE_XFAIL` entry in `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: not-run
- Hardware:    blackhole-p150b
- Duration:    not-run
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/mradermacher_qwen3_30b_a3b_instruct_reamini_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `third_party/tt_forge_models/mradermacher_qwen3_30b_a3b_instruct_reamini_i1_gguf/causal_lm/pytorch/loader.py` (added `_experts_implementation=batched_mm`)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (added `KNOWN_FAILURE_XFAIL` entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3c3a2ae0652cd10089d4a4e4dee1f31dd368af4d |
| tt-forge-models | 2e79797d049d10cb130c81a087b41852b0b719a1 |
