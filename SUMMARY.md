# Remediation Summary: qwen_3_30b_a3b_thinking_claude_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_Claude_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_30b_a3b_thinking_claude_gguf/causal_lm/pytorch-30B_A3B_Thinking_2507_Claude_GGUF-single_device-inference]

## Result
XFAIL — 30B-param MoE model dequantized to BF16 requires ~60 GB, exceeding p150b 32 GB DRAM; RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 (OOM)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-30b-moe-bf16-exceeds-p150b-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault (original); after loader fixes: RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Two loader bugs and a hardware capacity ceiling:

1. **Session contamination (loader):** 26 other GGUF loaders in the test suite patch `load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)`. When pytest collects the full test suite, these patches run at import time and replace the global `load_gguf_checkpoint`. Transformers 5.2.0 added a `model_to_load` kwarg to this call site, which the narrow-sig wrappers do not accept, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fix: widen all 26 signatures to accept `**kwargs` and pass them through.

2. **Qwen3MoE for-loop segfault (loader):** Qwen3-30B-A3B is a 128-expert MoE model. With the default `_experts_implementation` falling back to `"eager"` (Python for-loop) when `grouped_mm` is unavailable, the expert dispatch produces dynamic shapes that crash `partition_fx_graph_for_cpu_fallback` in torch_xla during Dynamo tracing. Fix: pass `experts_implementation="batched_mm"` to `from_pretrained`, which uses a static batched matrix-multiply for expert routing.

3. **Hardware capacity (terminal):** After fixing both loader bugs, the model compiles for ~24 minutes and then fails with INTERNAL: Error code: 13 (OOM) at `_run_cached_graph`. Qwen3-30B-A3B has 30B total parameters across 48 layers × 128 experts. Loaded with `torch_dtype=bfloat16`, the dequantized model requires ~60 GB DRAM (30B params × 2 bytes), which exceeds the p150b (Blackhole) single-device limit of 32 GB. This matches the `EXCLUDE_MODEL` disposition already applied to the non-GGUF `qwen_3/causal_lm/pytorch-30B_A3b`.

## Fix
- **tt_forge_models** `remediation/qwen_3_30b_a3b_thinking_claude_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_Claude_GGUF-single_device-inference`:
  - `**/causal_lm/pytorch/loader.py` (26 files): Widened `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` → `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs)` and updated the inner `_orig_load_gguf_checkpoint` call to pass `**kwargs`.
  - `qwen_3_30b_a3b_thinking_claude_gguf/causal_lm/pytorch/loader.py`: Added `model_kwargs.setdefault("experts_implementation", "batched_mm")` to use static expert dispatch.
- **tt-xla** `remediation/qwen_3_30b_a3b_thinking_claude_gguf-causal_lm-pytorch-30B_A3B_Thinking_2507_Claude_GGUF-single_device-inference`:
  - `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test.

## Verification
- pytest exit: FAIL (INTERNAL: Error code: 13 — OOM, hardware-class)
- Hardware:    blackhole-p150b
- Duration:    1466.55s (0:24:26) — compilation time before OOM at execution
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/qwen_3_30b_a3b_thinking_claude_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/<26 other GGUF loaders>/causal_lm/pytorch/loader.py` (narrow-sig fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 09ec11a61947fbd2f4484459c94b645bbee7e7ee |
| tt-forge-models | 95f58d84b76e6d5db4b9a605083509b9dbf77e29 |
