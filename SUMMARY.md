# Remediation Summary: deepseek-deepseek_moe-pytorch-16B_Base-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_moe/pytorch-16B_Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
deepseek-moe-data-dependent-dispatch-nan

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.95.

## Root cause
Loader layer. `DeepseekMoE.moe_infer` (inference-time expert dispatch) contains
data-dependent operations incompatible with XLA/torch.compile static-graph tracing:

1. `flat_expert_indices.bincount().cpu().numpy().cumsum(0)` — forces a host-device
   transfer mid-graph and uses NumPy, causing incorrect outputs on TT hardware.
2. A loop over `tokens_per_expert` slices whose lengths depend on runtime routing
   decisions — data-dependent control flow that cannot be statically traced.
3. `scatter_reduce_` with data-dependent indices — unsupported on TT.

Together these cause the compiled model to produce all-NaN outputs, failing PCC.

## Fix
Added `_deepseek_moe_forward` in `tt-xla:python_package/tt_torch/torch_overrides.py`.
This replaces `DeepseekMoE.forward` on the device path with a dense bmm over all
64 experts simultaneously (static shapes, torch.compile-friendly):

- Stack expert weights: `gate_w/up_w [E, H, inter]`, `down_w [E, inter, H]`
- Replicate all tokens: `[T, H] → [E, T, H]` via `expand().contiguous()`
- Compute all expert outputs: 2× bmm + act_fn + 1× bmm → `out_all [E, T, H]`
- Build dense routing weights `[T, E]` via `F.one_hot(topk_idx, E) * topk_weight`
- Weighted sum over experts → final output `[T, H]`

CPU path retains the original `moe_infer` loop for golden reference (PCC comparison).

The patch is applied in `tt-forge-models:deepseek/deepseek_moe/pytorch/loader.py`
after `AutoModelForCausalLM.from_pretrained`, walking modules to find `DeepseekMoE`
and patching the class (all layers share the same class, so one patch covers all).

This is not a forbidden workaround: the full model runs on device with no trimming,
CPU offload, or threshold change.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    ~22 minutes (model load + compilation + inference)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `_deepseek_moe_forward`
- `tt-forge-models/deepseek/deepseek_moe/pytorch/loader.py` — apply patch post-load

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 68ce3decad2aab99180e49cf01cbda6a12ed1bda |
| tt-forge-models | 951014b4c5b57ade3e7dae0d22d9e3d8ab4e9bff |
