# Remediation Summary: granitemoehybrid-causal_lm-pytorch-tiny_random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granitemoehybrid/causal_lm/pytorch-tiny_random-single_device-inference]

## Result
SILICON_PASS — loader fix applied; MoE expert dispatch no longer triggers device-to-host PJRT transfer

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full traceback origin:
  transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:1215:
    hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
  → GraniteMoeHybridParallelExperts.forward (inputs.split(expert_size))
  → GraniteMoeHybridTopKGating.forward (expert_size.tolist())
  → torch_xla.sync → torch_xla._XLAC._xla_step_marker
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
`GraniteMoeHybridTopKGating.forward` in transformers computes `expert_size`
as a device tensor and then calls `.tolist()` on it to build a Python list
used to split inputs across experts.  On TT silicon, calling `.tolist()` on
an XLA device tensor triggers a device-to-host PJRT transfer that fails with
gRPC INTERNAL error code 13.  The `GraniteMoeHybridParallelExperts.forward`
method then tries to call `inputs.split(expert_size)` with that list, which
never executes because the tolist() already crashed.

The fix belongs in the loader layer — patch both modules at load time to
avoid any device→host transfers and keep all dispatch logic in tensor
operations.

## Fix
Three commits applied to `tt_forge_models` submodule
(granitemoehybrid/causal_lm/pytorch/loader.py):

1. `5529f70747` — Add `_patched_topk_gating_forward` that returns
   `sorted_expert_ids` as an int32 tensor instead of calling `.tolist()`;
   add `_patched_parallel_experts_forward` using weight gather + einsum.

2. `f8bf620189` — Fix L1 CB overflow from the gather approach:
   `self.weight[sorted_expert_ids]` caused MLIR to flatten the weight to a
   2D embedding table whose row (~3 MB) overflowed the 1.5 MB L1 CB budget.
   Replaced with matmul + one-hot masking on the full weight tensor.

3. `8872f513ac` — Fix PCC regression from the one-hot matmul: the large
   `[T, num_experts*output_size]` intermediate degraded BF16 precision
   (PCC=0.93 vs 0.99 required).  Replaced with a statically-unrolled
   per-expert loop using `F.linear` with boolean masking — constant-indexed
   weight slices, no dynamic gather, same numerical path as reference.

`_patch_moe_experts(model)` is called immediately after
`AutoModelForCausalLM.from_pretrained` in `ModelLoader.load_model`.

tt_forge_models submodule updated from `0f7b734348` to `8872f513ac`.
tt-xla remediation branch: `remediation/granitemoehybrid-causal_lm-pytorch-tiny_random-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    55.01s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models` (submodule pointer update)
- `granitemoehybrid/causal_lm/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 86ac06cd40facabd233608f68abcad31bc566e85 |
| tt-forge-models | 8872f513ac217967e88920d11ab89c1249a799b0 |
