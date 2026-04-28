# Remediation Summary: llama_4_scout_bnb_4bit-causal_lm-pytorch-17B_16E_Instruct_unsloth_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_4_scout_bnb_4bit/causal_lm/pytorch-17B_16E_Instruct_unsloth_bnb_4bit-single_device-inference]

## Result
XFAIL — Llama 4 Scout 17B-16E is a 109B total-parameter MoE model; at NF4 4-bit quantization the weights are ~57 GB, exceeding the 24 GB p150b DRAM

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-llama4-scout-109b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Local reproduction (bitsandbytes not installed in venv):
```
E   RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
    model.layers.{0...47}.feed_forward.experts.down_proj  | MISSING
    model.layers.{0...47}.feed_forward.shared_expert.down_proj.weight | MISMATCH | ckpt: torch.Size([20971520, 1]) vs model: torch.Size([5120, 8192])
```

CI failure (with bitsandbytes installed):
```
E   RuntimeError: TT_THROW @ /home/ttuser/hf-bringup/tt-xla/pjrt_implementation/src/api/buffer_instance.cc:282: tt::exception
```
buffer_instance.cc:282 = `TT_THROW("Complex tensor with num_dims == 0 is not supported.")`

## Root cause
**Hardware capacity**: Llama 4 Scout 17B-16E (`unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit`) is a mixture-of-experts model with 109B total parameters across 16 experts (48 layers, hidden_size=5120). At NF4 4-bit quantization the weights occupy ~57 GB on disk, well above the 24 GB device DRAM on p150b.

**Local failure mechanism**: `bitsandbytes` was absent from the loader's `requirements.txt`. Without it, `AutoModelForCausalLM.from_pretrained` ignores the `quantization_config` and tries to load in full precision. The checkpoint stores weights in BNB 4-bit packed format (e.g. `[20971520, 1]` uint8 instead of `[5120, 8192]` bf16), so `from_pretrained` raises a size-mismatch error before the model even reaches the device.

**CI failure mechanism**: With bitsandbytes available the model loads in 4-bit format. During torch.compile/XLA compilation the Llama 4 text RoPE (`Llama4TextRotaryEmbedding.forward`) calls `torch.polar(...)` and returns a `complex64` tensor. A 0-dimensional complex scalar appears somewhere in the compilation pipeline and the TT PJRT backend raises at `buffer_instance.cc:282: TT_THROW("Complex tensor with num_dims == 0 is not supported.")`. Even if this compilation error were fixed, the model would OOM on device (57 GB 4-bit > 24 GB DRAM; after dequantization to bf16 the weight footprint reaches ~218 GB).

An earlier remediation attempt (`4fc6d1b827` in tt-forge-models) used forbidden workarounds (trimming to 6 layers, replacing hidden_size with 1024, loading via `from_config` with random weights). That attempt is treated as evidence the bug is unfixed.

## Fix
1. **Loader fix** (`tt_forge_models`, branch `remediation/llama_4_scout_bnb_4bit-...`):
   Added `bitsandbytes>=0.46.1` to `llama_4_scout_bnb_4bit/causal_lm/pytorch/requirements.txt` so the model loads via the quantization_config path and fails with the correct device-side error rather than a loader mismatch.

2. **Test config** (`tt-xla`, branch `remediation/llama_4_scout_bnb_4bit-...`):
   Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: FAIL (local: ignore_mismatched_sizes before bitsandbytes fix; CI: TT_THROW at buffer_instance.cc:282)
- Hardware:    p150b (blackhole)
- Duration:    2179.44s (0:36:19) for local reproduction (model load dominates; fails at loading stage)
- Tier A attempts: N/A

## Files changed
- `llama_4_scout_bnb_4bit/causal_lm/pytorch/requirements.txt` (new, in tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 50e20d831fd3d8600bc47cfc420936ecec007c62 |
| tt-forge-models | 29db0fba93ff91aa7b8ab3ce211c04ff6e857f7b |
