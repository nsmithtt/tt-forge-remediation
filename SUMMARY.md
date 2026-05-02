# Remediation Summary: minerva_moe_2x3b-causal_lm-pytorch-Minerva-MoE-2x3B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minerva_moe_2x3b/causal_lm/pytorch-Minerva-MoE-2x3B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
N/A

## Bug fingerprint
grouped-mm-cpu-fallback-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

Crash location (from traceback):
  torch_overrides.py:34 in __torch_function__
  partition_fx_graph_for_cpu_fallback (dynamo_bridge.py:762)

## Root cause
`MixtralExperts.forward` (used by Minerva MoE 2x3B — a MixtralForCausalLM variant) has two problems that prevent running on TT device:

1. **Segfault**: The default eager path uses `nonzero()` + a Python for-loop with a data-dependent trip count. During `partition_fx_graph_for_cpu_fallback`, the FX interpreter runs ops through `TorchFunctionOverride.__torch_function__`, and calling `nonzero()` on a TT-placed tensor triggers an unsupported device-to-host path, causing a segfault.

2. **OOM / no lowering**: Two alternative implementations also fail — `batched_mm` materialises a ~60 GB intermediate (gate_up for all experts × all tokens × all experts), and `_grouped_mm` has no TT lowering.

The fix is the same pattern already applied for `GptOssExperts` and `Qwen3MoeExperts`: replace `forward` with a device-friendly dense einsum over all experts (CPU path keeps the per-expert loop for PCC correctness).

## Fix
Cherry-picked two commits from `origin/remediation/mixtral_8x7b_moe_rp_story_gguf-causal_lm-pytorch-8x7B_MoE_RP_Story_GGUF-single_device-inference` into a new `remediation/minerva_moe_2x3b-causal_lm-pytorch-Minerva-MoE-2x3B-single_device-inference` branch in `tt-xla`:

- `431bec5a2` — add `_mixtral_experts_forward` in `python_package/tt_torch/torch_overrides.py`:
  - CPU path: original per-expert loop (PCC golden reference)
  - Device path: `torch.einsum` over all experts (`[E,T,H] × [E,2I,H]` → activate → `[E,T,I] × [E,H,I]`) followed by routing weight aggregation; no `nonzero`, no `histc`, no `_grouped_mm`
  - Monkey-patches `MixtralExperts.forward` at import

- `1d6cc564f` — switch device path from `bmm+permute` to `einsum` to avoid materialising a 1.75 GB contiguous transposed copy of weights that would OOM p150b DRAM after model load.

File changed: `python_package/tt_torch/torch_overrides.py`

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 87.46s
- Tier A attempts: N/A

## Files changed
- tt-xla: `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1d6cc564f29bf51f0084353e0680c276a0b60faa |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
