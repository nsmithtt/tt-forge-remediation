# Remediation Summary: brown_loafers_70b_i1_gguf-causal_lm-pytorch-BrownLoafers_70B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[brown_loafers_70b_i1_gguf/causal_lm/pytorch-BrownLoafers_70B_i1_GGUF-single_device-inference]

## Result
XFAIL — 70B model dequantized to BF16 (~141 GB) exceeds single-device DRAM capacity (~32 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After loader fix, failure during silicon execution:
```
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks,
where each bank needs to store 58720256 B, but bank size is 4273390016 B
(allocated: 4113318080 B, free: 160071936 B, largest free block: 45351360 B)
```

## Root cause
Two issues:

**Loader (ImportError):** `gguf>=0.10.0` was not listed in the model's
`requirements.txt`. The test environment did not have `gguf` installed, so
`is_gguf_available()` returned False inside transformers'
`modeling_gguf_pytorch_utils.py`, raising the ImportError before any model
loading could begin. A secondary issue was missing chat-template fallback in
`load_inputs`.

**Hardware capacity (OOM):** The BrownLoafers 70B i1 GGUF model
(mradermacher/BrownLoafers-70B-i1-GGUF, `BrownLoafers-70B.i1-Q4_K_M.gguf`,
42.5 GB on disk) is a LLaMA architecture model with 80 layers, hidden_size=8192,
intermediate_size=28672. When loaded by transformers via `from_pretrained` with
`gguf_file=`, weights are dequantized to BF16, yielding ~141 GB of model weights.
The single TT device has ~32 GB of DRAM (8 banks × ~4 GB). Even at Q4_K_M
quantization (~42.5 GB), the model exceeds device DRAM capacity. During the
silicon run, ~30.6 GB was allocated for model weights before an OOM prevented
tilizing the next tensor due to memory fragmentation.

## Fix
**Loader fix (tt_forge_models remediation branch):**
- `brown_loafers_70b_i1_gguf/causal_lm/pytorch/requirements.txt`: added `gguf>=0.10.0`
- `brown_loafers_70b_i1_gguf/causal_lm/pytorch/loader.py`: added chat-template
  fallback in `load_inputs` (try/except ValueError → use raw sample_text)

**Test config (tt-xla remediation branch):**
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  added `KNOWN_FAILURE_XFAIL` entry with reason explaining hardware capacity limit

## Verification
- pytest exit: FAIL (OOM after loader fixes)
- Hardware:    n150
- Duration:    1664.70s (0:27:44) until OOM
- Tier A attempts: N/A

## Files changed
- `tt_forge_models: brown_loafers_70b_i1_gguf/causal_lm/pytorch/requirements.txt`
- `tt_forge_models: brown_loafers_70b_i1_gguf/causal_lm/pytorch/loader.py` (load_inputs fallback)
- `tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 784a810566c632024c9ce48cc925f32d947d8839 |
| tt-forge-models | 6ee97dd6f3f1aabe041e6a94721c5cd4ebd1a75b |
