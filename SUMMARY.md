# Remediation Summary: deepseek-deepseek_r1_distill_bnb-pytorch-Distill_Qwen_32B_BNB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_distill_bnb/pytorch-Distill_Qwen_32B_BNB_4bit-single_device-inference]

## Result
XFAIL — 32B params dequantized to BF16 requires ~64 GB; n150 DRAM is ~32 GB (hardware-class capacity ceiling)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
bnb-dequantize-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

## Root cause
Two loader bugs prevent the test from progressing to silicon, and after fixing both, the
model is a hardware-class XFAIL:

**Loader bug 1 — missing bitsandbytes dependency**: The `Distill_Qwen_32B_BNB_4bit`
variant loads `unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit`, a BNB 4-bit quantized
checkpoint that requires `bitsandbytes>=0.46.1`. No `requirements.txt` existed for
this loader directory; transformers raises ImportError at model load time.

**Loader bug 2 — Params4bit incompatible with TT device**: BNB 4-bit quantization
stores weights as `Params4bit` tensors, which are CUDA-only. The test framework calls
`model.to(tt_device)`, which fails because `Params4bit.detach()` returns a plain
`Tensor`, not a `Params4bit`. Fix: dequantize all `Linear4bit` modules to regular
`nn.Linear` (bfloat16) before returning the model.

**Hardware capacity ceiling**: After dequantization, the 32B parameter model requires
approximately 32 × 10⁹ × 2 bytes ≈ 64 GB in bfloat16. The n150 device has
approximately 32 GB of DRAM (consistent with the 70B LLaMA GGUF OOM observed in
other reports). 64 GB >> 32 GB; the model cannot fit on a single n150. This is not
a compiler bug — it is a genuine hardware capacity limitation. The same BNB 4-bit
→ dequantize → device transfer pattern was validated on the 7B variant
(`deepseek-deepseek_r1_distill-pytorch-Distill_Qwen_7B_unsloth_bnb_4bit`), which
dequantizes to ~14 GB and fits on n150.

## Fix
**Loader fixes (committed to tt-forge-models remediation branch
`remediation/deepseek-deepseek_r1_distill_bnb-pytorch-Distill_Qwen_32B_BNB_4bit-single_device-inference`):**

1. `deepseek/deepseek_r1_distill_bnb/pytorch/requirements.txt` (new file):
   Added `bitsandbytes>=0.46.1`.

2. `deepseek/deepseek_r1_distill_bnb/pytorch/loader.py`: Added
   `_dequantize_bnb_model()` static method that iterates through all `Linear4bit`
   modules and replaces each with a dequantized `nn.Linear`. Uses
   `bitsandbytes.functional.dequantize_4bit(weight.data, quant_state)` to recover
   float weights, then converts to bfloat16. Called in `load_model()` after
   `from_pretrained`.

**Test config update (committed to tt-xla remediation branch
`remediation/deepseek-deepseek_r1_distill_bnb-pytorch-Distill_Qwen_32B_BNB_4bit-single_device-inference`):**

3. `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
   Added `deepseek/deepseek_r1_distill_bnb/pytorch-Distill_Qwen_32B_BNB_4bit-single_device-inference`
   with `status: KNOWN_FAILURE_XFAIL` and an explanatory reason string.

## Verification
- pytest exit: FAIL (ImportError before fixes; hardware XFAIL after fixes — not run on silicon)
- Hardware:    n150
- Duration:    not-run
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_distill_bnb/pytorch/requirements.txt` (new, in tt-forge-models)
- `deepseek/deepseek_r1_distill_bnb/pytorch/loader.py` (modified, in tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (modified, in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7cf3b341a2e3845826a77cf9b98092e1b2f0c5af |
| tt-forge-models | 0953857ae07616b3b45210134465cc6d08783767 |
