# Remediation Summary: mixtao-causal_lm-pytorch-7Bx2_MoE_v8_1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mixtao/causal_lm/pytorch-7Bx2_MoE_v8.1-single_device-inference]

## Result
SILICON_PASS — static per-expert masked matmul replaced grouped_mm dispatch; test passes in 324s on p150b

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mixtral-grouped-mm-histc-on-int-expert-dispatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Segfault in partition_fx_graph_for_cpu_fallback during XLA probing, triggered
by torch.histc being called on an int tensor in grouped_mm_experts_forward:

    While executing %histc : call_function[target=torch.ops.aten.histc.default]
        num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, ...)

## Root cause
The MixTAO-7Bx2-MoE-v8.1 model (Mixtral architecture, num_experts=2,
num_experts_per_tok=2) has config._experts_implementation="grouped_mm" set in
its config.json. The transformers `@use_experts_implementation` decorator on
MixtralExperts dispatches the forward to `grouped_mm_experts_forward`, which
calls `torch.histc(expert_ids.int(), bins=num_experts, ...)`. On TT XLA, this
int tensor requires a device-to-host transfer during partition_fx_graph_for_cpu_fallback
probing, causing a segfault (INTERNAL: Error code: 13).

The fix is a static per-expert masked matmul loop over range(num_experts), which:
1. Avoids torch.histc entirely
2. Uses no nonzero() or dynamic shapes
3. Computes expert routing weights as a static mask: (top_k_index == expert_idx)
4. Works trivially when num_experts_per_tok == num_experts (all experts always active)

## Fix
Added `_patch_mixtral_experts_static(model)` call in `load_model()` in:
  `mixtao/causal_lm/pytorch/loader.py`

The patch replaces `MixtralExperts.forward` on all expert instances with a
static implementation that iterates `for expert_idx in range(self.num_experts)`
and computes routing via `(top_k_weights * (top_k_index == expert_idx).to(weight_dtype)).sum(-1)`.
Explicit dtype casts to `gate_up_proj.dtype` (bfloat16) handle the float32
top_k_weights that the Mixtral gate outputs by default.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    324.31s (0:05:24)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/mixtao/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | c7dcd3891 |
| tt-forge-models | d0200a717d |
