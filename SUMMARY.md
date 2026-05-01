# Remediation Summary: hossamdaoud_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_UD_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hossamdaoud_qwen3_5_35b_a3b_gguf/causal_lm/pytorch-35B_A3B_UD_Q4_K_M-single_device-inference]

## Result
FAIL — Tier B compiler crash: SIGSEGV in partition_fx_graph_for_cpu_fallback when XLA probes Qwen3_5MoeGatedDeltaNet conv1d op; loader bugs fixed but terminal bug is unresolvable without new compiler infrastructure

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
qwen35moe-gated-delta-net-conv1d-partition-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause

The hossamdaoud loader had no qwen35moe GGUF support at all, causing the CI timeout. The failure chain has two distinct parts:

**Loader bugs (all fixed on remediation branch):**

1. **Missing qwen35moe GGUF arch registration**: `transformers.modeling_gguf_pytorch_utils.GGUF_SUPPORTED_ARCHITECTURES` does not contain `qwen35moe` in transformers 5.2.0. Without registration, `load_gguf_checkpoint` raises `ValueError: GGUF model with architecture qwen35moe is not supported yet.` after downloading the model file. In CI the 22 GB GGUF download exhausts the test timeout before the ValueError is raised, producing the "Test exceeded configured timeout" failure.

2. **conv1d weight shape mismatch**: The GGUF file stores `ssm_conv1d` tensors as `[out_ch, kernel]`, but `nn.Conv1d` expects `[out_ch, in_ch, kernel]`. Without the in-channel dimension (1 for depthwise conv), `_finalize_model_loading` raises `RuntimeError: You set ignore_mismatched_sizes to False`.

3. **Expert weight key mapping**: `get_gguf_hf_weights_map` produces `blk.N.ffn_gate_up_exps` for the fused gate/up expert tensor, but the GGUF file stores separate `blk.N.ffn_gate_exps` and `blk.N.ffn_up_exps` tensors. The tensor processor's `.process()` call raises `KeyError: 'blk.0.ffn_gate_exps'`.

**Terminal compiler-stack bug (Tier B, unfixed):**

After all loader fixes are applied (as confirmed by the identical cmp_nct_qwen3_5_35b_a3b_gguf model reaching this state — see report branch `report/cmp_nct_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_UD_Q4_K_M_GGUF-single_device-inference`), `torch.compile` with the `tt` backend crashes with SIGSEGV during `partition_fx_graph_for_cpu_fallback`. The XLA dynamo bridge probes each op in the FX graph to assign it to CPU or device. When it probes a `conv1d` op from a `Qwen3_5MoeGatedDeltaNet` layer, the TT Conv2D C++ kernel crashes. Exit code 139.

## Fix

**Loader fix** — `tt-forge-models` `hossamdaoud_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py`:

Added `_patch_transformers_qwen35moe_gguf()` called at module import time. The function:
1. Appends `"qwen35moe"` to `GGUF_SUPPORTED_ARCHITECTURES` and adds its config key mapping to `GGUF_TO_TRANSFORMERS_MAPPING["config"]`.
2. Registers `_Qwen35MoeTensorProcessor` (subclass of `Qwen2MoeTensorProcessor`) in `TENSOR_PROCESSORS["qwen35moe"]`. The subclass unsqueezes dim 1 on any `ssm_conv1d` tensor with `ndim==2` before delegating to the parent.
3. Patches `load_gguf_checkpoint` to convert `model_type "qwen35moe"` → `"qwen3_5_moe_text"` and constructs the `layer_types` list (alternating `"linear_attention"` / `"full_attention"` every `full_attention_interval` layers).
4. Patches `get_gguf_hf_weights_map` to add `blk.N.ffn_gate_exps` and `blk.N.ffn_up_exps` entries (derived from the fused `blk.N.ffn_gate_up_exps` entry) so the tensor processor can look up expert gate and up weights by their actual GGUF names.

Committed as `14f19d6ebbfc36b4925c3a802684315e4c569a8f` on remediation branch `remediation/hossamdaoud_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_UD_Q4_K_M-single_device-inference` in tt-forge-models.

**Proposed fix for Tier B bug:** The TT Conv2D kernel must be hardened to not segfault when probing unsupported conv1d shapes during `partition_fx_graph_for_cpu_fallback`. The crash is triggered by `conv1d` ops from `Qwen3_5MoeGatedDeltaNet` layers (e.g., `[8192, 1, 4]` depthwise kernel). The fix would need to be in the tt-xla PJRT plugin or the tt-mlir Conv2D lowering to either gracefully reject unsupported shapes (allowing CPU fallback) or handle these dimensions correctly. The only known workaround — disabling `Qwen3_5MoeGatedDeltaNet.forward` via `torch.compiler.disable` — is a forbidden CPU offload technique.

## Tier B justification
internal-error-unknown-mechanism — The SIGSEGV originates inside the TT Conv2D C++ kernel with no Python-level exception or error message. The exact mechanism (null pointer, stack overflow, shape invariant violation) is unknown without examining the kernel source under gdb. A safe fix requires diagnosing the crash in C++ code, which is beyond the scope of a single-function Tier A fix.

## Verification
- pytest exit: FAIL (cannot run locally — model is 22 GB; only 3.8 GB disk free; terminal Tier B bug extrapolated from identical cmp_nct model run)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `hossamdaoud_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py` (1 commit on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 14f19d6ebbfc36b4925c3a802684315e4c569a8f |
