# Remediation Summary: gpt_oss_20b_counsel_mindbuddi_gguf-causal_lm-pytorch-20B_Counsel_MindBuddi_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_counsel_mindbuddi_gguf/causal_lm/pytorch-20B_Counsel_MindBuddi_GGUF-single_device-inference]

## Result
XFAIL — 20B MoE model dequantized to BF16 (~40 GB) exceeds single-device DRAM (~34 GB on p150b); three loader bugs fixed along the way

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-dram-oom-20b-moe-bf16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `raise AttributeError(`

Reproduced as three sequential bugs:

1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
2. `ValueError: The checkpoint you are trying to load has model type 'gpt-oss' but Transformers does not recognize this architecture.`
3. `AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'` in `load_shard_spec`
4. Segfault in `partition_fx_graph_for_cpu_fallback` (Qwen3MoeExperts.forward data-dependent nonzero)
5. `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` in `_run_cached_graph` (DRAM OOM)

## Root cause
**Loader bugs (3 fixed):**

1. **model_to_load TypeError**: Other loaders (e.g., `bartowski_coniccat_qwen3_5_27b_writer_gguf`) patch `load_gguf_checkpoint` at import time with a 2-arg signature missing `model_to_load`. pytest imports all loaders during collection; the broken patch is active when this model's `from_pretrained` runs. Fix: context manager `_gguf_for_gpt_oss()` installs a correctly-signed patcher on all 4 binding sites, recovers the True Original via `sys.modules` search for `_orig_load_gguf_checkpoint`.

2. **gpt-oss architecture unrecognized**: `gpt-oss` is not in transformers' `CONFIG_MAPPING` or `GGUF_SUPPORTED_ARCHITECTURES`. Fix: `_patch_gpt_oss_support()` registers `gpt-oss` as alias for `qwen3_moe` in all GGUF tables; `_patched` remaps `model_type: "gpt-oss"` → `"qwen3_moe"` in the returned config.

3. **load_shard_spec wrong MLP shape**: Original assumed dense LLaMA-style MLP (`up_proj`/`gate_proj`/`down_proj`). GPT-OSS 20B is Qwen3MoE with `Qwen3MoeSparseMoeBlock` which has `mlp.experts.gate_up_proj` (fused, `[E, 2*I, H]`) and `mlp.experts.down_proj` (`[E, H, I]`). Fix: guard with `hasattr(mlp, "experts")` and use 3-tuple shard dims `(None, "model", "batch")`.

**Compiler-stack fix (Tier A, in tt-xla):**

4. **Qwen3MoeExperts segfault**: `Qwen3MoeExperts.forward` uses data-dependent `nonzero()` + for-loop to dispatch tokens to experts. During `partition_fx_graph_for_cpu_fallback`, the XLA FX interpreter traces this code with XLA tensors while `TorchFunctionMode` is globally active, causing a segfault. Fix: monkey-patch `Qwen3MoeExperts.forward` in `torch_overrides.py` — CPU path uses original per-expert loop; device path uses static dense batched matmul (expand + bmm + gather + weighted sum, no data-dependent shapes).

**Hardware-class ceiling:**

5. **DRAM OOM**: After all fixes the compiled graph was executed. Error code: 13 occurred at `to_device` within `_run_cached_graph`, confirming that model weights could not be loaded to device DRAM. The 20B MoE model has 24 transformer layers with 32 experts per layer; dequantized to BF16 the weight tensor set is ~40 GB, exceeding the p150b's ~34 GB DRAM.

## Fix
**tt_forge_models** — `gpt_oss_20b_counsel_mindbuddi_gguf/causal_lm/pytorch/loader.py`:
- Rewrote with `_find_true_load_gguf_checkpoint()` (sys.modules search), `_patch_gpt_oss_support()` (GGUF table registration), and `_gguf_for_gpt_oss()` context manager (fixes model_to_load TypeError + model_type remapping)
- Updated `load_shard_spec` to handle `Qwen3MoeSparseMoeBlock` via `hasattr(mlp, "experts")` guard

**tt-xla** — `python_package/tt_torch/torch_overrides.py`:
- Added `_qwen3moe_experts_forward()` with device-friendly dense bmm path (no data-dependent shapes)
- Registered patch on `Qwen3MoeExperts.forward` via try/import block

**tt-xla** — `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
- Added `KNOWN_FAILURE_XFAIL` entry for `gpt_oss_20b_counsel_mindbuddi_gguf/causal_lm/pytorch-20B_Counsel_MindBuddi_GGUF-single_device-inference`

## Verification
- pytest exit: FAIL (DRAM OOM → XFAIL disposition)
- Hardware:    blackhole-p150b
- Duration:    1170.25s (0:19:30)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models/gpt_oss_20b_counsel_mindbuddi_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 28f4142653018199dff4a433fd17b8d02dc1dbca |
| tt-forge-models | e3853b1d9a76c0eb739d5e6df79fbe1f0eb4f2a4 |
