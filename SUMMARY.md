# Remediation Summary: cmp_nct_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_UD_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cmp_nct_qwen3_5_35b_a3b_gguf/causal_lm/pytorch-35B_A3B_UD_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — SIGSEGV in partition_fx_graph_for_cpu_fallback when XLA probes GatedDeltaNet conv1d op on TT device; Tier B compiler-stack bug with no safe fix available

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
Original reported failure: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`

Actual failure sequence on the target branch after reproduction:
1. `KeyError: 'blk.0.ffn_gate_exps'` — GGUF gate/up tensor key missing from mapping
2. After fix 1: `RuntimeError: You set ignore_mismatched_sizes to False` — conv1d.weight shape mismatch (ckpt: [8192,4] vs model: [8192,1,4])
3. After fix 2: SIGSEGV (exit code 139) in `partition_fx_graph_for_cpu_fallback` when XLA probes the `conv1d` op in Qwen3.5 MoE GatedDeltaNet layers

## Root cause

**Loader bug 1 (fixed):** `patched_get_gguf_hf_weights_map` in the cmp_nct loader translated `qwen3_5_moe_text` → `qwen35moe` but did not add the separate `ffn_gate_exps`/`ffn_up_exps` entries that the GGUF file uses. `gate_up_proj` is an `nn.Parameter` (no `.weight` suffix), so the weight map only contained `blk.N.ffn_gate_up_exps` (the combined HF key). The GGUF checkpoint stores separate `ffn_gate_exps` and `ffn_up_exps` tensors, so `process()` failed with `KeyError: 'blk.0.ffn_gate_exps'`. Same root cause as fixed in `amarck`/`claude_opus` loaders by commit `5358642568` on branch `arch-c-36-tt-xla-dev/nsmith/2026-04-27/qwen35moe-gguf-fix`.

**Loader bug 2 (fixed):** The loader set `TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]`, reusing `Qwen2MoeTensorProcessor` directly. Qwen3.5 MoE's GatedDeltaNet layers have `ssm_conv1d` tensors that the GGUF stores as `[out_ch, kernel]`, but `nn.Conv1d` expects `[out_ch, in_ch, kernel]` (depthwise, `in_ch=1`). The processor did not unsqueeze dim 1, causing a shape mismatch (`[8192,4]` vs `[8192,1,4]`). Fixed by creating `_Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor)` that unsqueezes 2D conv1d weights.

**Compiler-stack bug (unfixed):** After the loader is fixed, `torch.compile` with the `tt` backend crashes with SIGSEGV during `partition_fx_graph_for_cpu_fallback`. The XLA dynamo bridge probes each op in the FX graph to assign it to CPU or device. When it probes a `conv1d` op from a `Qwen3_5MoeGatedDeltaNet` layer (the linear attention layers in this model), the TT Conv2D kernel segfaults. The crash is in `tt_torch/torch_overrides.py:__torch_function__` → `func(*args, **kwargs)` → TT kernel C++ crash. The only known workaround (`torch.compiler.disable` on `Qwen3_5MoeGatedDeltaNet.forward`) is a forbidden CPU offload technique.

## Fix

**Fix 1** — `tt-forge-models` `cmp_nct_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py`: In `patched_get_gguf_hf_weights_map`, changed `return orig_get_map(...)` to collect the result and add extra entries for `ffn_gate_exps` and `ffn_up_exps` by splitting on `.ffn_gate_up_exps` in each key. Committed on remediation branch `remediation/cmp_nct_qwen3_5_35b_a3b_gguf-causal_lm-pytorch-35B_A3B_UD_Q4_K_M_GGUF-single_device-inference` (commit `4e2504f911`).

**Fix 2** — Same file: Replaced `TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]` with a `_Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor)` subclass that unsqueezes dim 1 for `ssm_conv1d` tensors before delegating to `super().process()`. Class is stored in `TENSOR_PROCESSORS["qwen35moe"]` (not an instance, so transformers can call `ProcessorClass(config=config)`). Committed as `67047fb4bf` and corrected in `e2cb5737e2`.

**Proposed fix for Tier B bug:** The TT Conv2D kernel must be hardened to not segfault when probing for device placement during `partition_fx_graph_for_cpu_fallback`. The crash is triggered by `conv1d` ops with the specific tensor dimensions used in `Qwen3_5MoeGatedDeltaNet` (e.g., `[8192, 1, 4]` kernel). The fix would need to be in the tt-xla PJRT plugin or tt-mlir Conv2D lowering to either gracefully reject unsupported shapes (allowing CPU fallback) or handle these dimensions correctly.

## Tier B justification
internal-error-unknown-mechanism — The SIGSEGV originates inside the TT Conv2D C++ kernel with no Python-level exception or error message. The exact mechanism (null pointer, stack overflow, shape invariant violation) is unknown without running under gdb or examining the kernel source. A safe fix requires diagnosing the crash in C++ code.

## Verification
- pytest exit: FAIL (SIGSEGV, exit code 139)
- Hardware:    n150
- Duration:    ~25 minutes (model loading ~20 min + segfault during compilation)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `cmp_nct_qwen3_5_35b_a3b_gguf/causal_lm/pytorch/loader.py` (3 commits on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e2cb5737e244dfca11f284e52fdddd63ab81b273 |
