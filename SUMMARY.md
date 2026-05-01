# Remediation Summary: melinoe_qwen3_omni_gguf-causal_lm-pytorch-30B_A3B_Thinking_i1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[melinoe_qwen3_omni_gguf/causal_lm/pytorch-30B_A3B_Thinking_i1-single_device-inference]

## Result
XFAIL — 30B MoE model (~60 GB BF16) exceeds single-device p150b DRAM (~32 GB)

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-30b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: ImportError while loading conftest '/home/nsmith/hf-bringup/tt-xla/tests/conftest.py'.

Actual sequence of failures found and fixed:
1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`
2. `TorchRuntimeError: Dynamo failed to run FX node ... call_method scatter_` (in-place scatter_ rejected on FakeTensor)
3. `RuntimeError('scatter(): Expected self.dtype to be equal to src.dtype')` (routing weights float32 vs hidden bfloat16)
4. Terminal OOM: `Out of Memory: Not enough space to allocate 805306368 B DRAM buffer across 8 banks … (allocated: 4136282432 B, free: 137107584 B, largest free block: 50331648 B)`

## Root cause

Three separate issues:

**1. Loader bug — model_to_load kwarg dropped by global patcher chain**

At import time, 40+ loaders globally monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. These patches
do not forward the `model_to_load` keyword argument added in transformers 5.x.
Walking only module-level globals (`_orig_load_gguf_checkpoint`) misses patchers
that store their inner function as a closure variable (`orig_load`). Both traversal
paths are needed.

**2. tt-xla bug — Qwen3MoeExperts.forward data-dependent shapes crash compilation**

`Qwen3MoeExperts.forward` uses `nonzero()` + a Python for-loop, producing
data-dependent output shapes. This pattern crashes
`partition_fx_graph_for_cpu_fallback` in `torch_xla._dynamo.dynamo_bridge` when
tracing with XLA FakeTensors. Fixed in `torch_overrides.py` by replacing the
for-loop with a dense bmm over all experts (static-shape, compile-safe).
The initial bmm implementation had two sub-bugs: in-place `scatter_()` rejected by
XLA (fixed to `scatter()`), and dtype mismatch between routing weights (float32)
and the zeros tensor (bfloat16) for the scatter target.

**3. Hardware capacity ceiling**

The 30B Qwen3MoE model is loaded from Q4_K_M GGUF and dequantized to BF16 by
transformers. At BF16, 30B parameters require ~60 GB of device DRAM. The single
p150b device provides ~32 GB. At execution time 3.85/3.98 GB per bank was already
allocated by model weights; then a 768 MB tensor allocation failed. The same
architecture (qwen_3/30B_A3b) is already `EXCLUDE_MODEL: Too large for single
chip` in the test config.

## Fix

**Loader fix** (`tt-forge-models`):
Added `_unwrap_gguf_patcher()` + `_gguf_model_to_load_compat()` context manager
to `loader.py`. Traversal walks the full patch chain via both module-level
`_orig_load_gguf_checkpoint` globals and closure cells to reach the original
transformers function, then temporarily restores it for the duration of
`from_pretrained`. Also added `requirements.txt` with `gguf>=0.10.0`.

**tt-xla fix** (`torch_overrides.py`):
Added `_qwen3moe_experts_forward()` replacing `Qwen3MoeExperts.forward`.
CPU path retains the per-expert loop; device path uses dense bmm:
- out-of-place `scatter()` (XLA rejects in-place `scatter_()` on FakeTensors)
- `weights_flat.to(hidden_flat.dtype)` cast before scatter (routing weights are
  float32; hidden states are bfloat16 after model moves to bfloat16)

**Test config** (`test_config_inference_single_device.yaml`):
Added `KNOWN_FAILURE_XFAIL` entry for the hardware-class OOM.

## Verification
- pytest exit: FAIL (OOM — hardware-class, expected)
- Hardware:    blackhole-p150b
- Duration:    1528.00s (0:25:27) — third run, after all loader + tt-xla fixes applied
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/melinoe_qwen3_omni_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/melinoe_qwen3_omni_gguf/causal_lm/pytorch/requirements.txt`
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4f33ca31c55c5efe2aa588307d25ad5d78d414ea |
| tt-forge-models | f39c4fabea6d08c6e1da6796ebaa4cf7c88bb8f3 |
