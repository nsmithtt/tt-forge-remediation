# Remediation Summary: huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf/causal_lm/pytorch-30B_A3B_Thinking_2507_abliterated_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-30B-A3B BF16 dequantized from GGUF requires ~32.5 GB; p150b single-device DRAM is ~31.84 GB

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
followed by (after loader fix):
```
TT_FATAL: Out of Memory: Not enough space to allocate 805306368 B DRAM buffer across 8 banks,
where each bank needs to store 100663296 B, but bank size is 4273390016 B
(allocated: 4263102208 B, free: 10287808 B, largest free block: 4911552 B)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two bugs, one hardware-class ceiling:

1. **Loader bug — model_to_load TypeError**: transformers 5.2 added a
   `model_to_load` keyword argument to `load_gguf_checkpoint`. Other loaders in
   the test session (e.g., `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`) patch
   `load_gguf_checkpoint` at module-import time with a narrow signature
   `(gguf_path, return_tensors=False)` that does not accept `model_to_load`.
   Since those loaders are imported during pytest collection, the narrow patch
   is still active when our loader runs, causing the TypeError.

2. **Loader bug — Qwen3MoE segfault**: `Qwen3MoeExperts.forward()` uses a
   Python for-loop over a dynamically-sized `expert_hit` tensor that XLA/
   torch.compile cannot statically trace. This produces a segfault inside
   `partition_fx_graph_for_cpu_fallback`. Fix: set
   `model.config._experts_implementation = "batched_mm"` which uses only
   static tensor operations.

3. **Hardware-class OOM**: After fixing both loader bugs, the Qwen3-30B-A3B
   model loaded (579 GGUF tensors dequantized to BF16, ~32.5 GB) but OOMed
   during execution when trying to allocate one more DRAM buffer. The p150b
   device has ~31.84 GB total DRAM (8 banks × 3.98 GB), and the model needs
   ~32.5 GB. This is a genuine hardware capacity ceiling — the model is slightly
   too large for a single p150b chip in BF16.

## Fix
**tt-forge-models** (`remediation/huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_abliterated_GGUF-single_device-inference`):

- `huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf/causal_lm/pytorch/loader.py`:
  - Added `_patched_load_gguf_checkpoint(*args, **kwargs)` with wide signature to
    override narrow-sig contamination from other loaders; correctly forwards all
    arguments including `model_to_load` to the original function.
  - Added `model.config._experts_implementation = "batched_mm"` after `from_pretrained`
    to prevent the Qwen3MoE segfault in XLA graph partitioning.

**tt-xla** (`remediation/huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_abliterated_GGUF-single_device-inference`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
  Added `KNOWN_FAILURE_XFAIL` entry documenting the hardware OOM:
  `huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf/causal_lm/pytorch-30B_A3B_Thinking_2507_abliterated_GGUF-single_device-inference`

## Verification
- pytest exit: FAIL (OOM — hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    2453.12s (0:40:53) — includes 16+ min GGUF dequantization
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `huihui_qwen3_30b_a3b_thinking_2507_abliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1049ea1e190751329f86f65817f9ef2d9b08a212 |
| tt-forge-models | 00a09b205feb48164130a5a2e1c3ca53e021bf9a |
