# Remediation Summary: granitemoehybrid-causal_lm-pytorch-granite_4_0_h_tiny-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granitemoehybrid/causal_lm/pytorch-granite_4_0_h_tiny-single_device-inference]

## Result
FAIL — loader fix resolves INTERNAL error 13; PCC=0.907 < 0.99 on Blackhole from Mamba SSM BF16 precision (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
mamba-ssm-bf16-pcc-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
at torch_xla._XLAC._xla_step_marker → torch_xla.sync → bridge.extract_compiled_graph →
partition_fx_graph_for_cpu_fallback → extract_internal → extract_graph_helper

(After loader fix: PCC comparison failed. Calculated: pcc=0.906953025589983. Required: pcc=0.99.)

## Root cause
**Two bugs:**

1. **Loader bug (fixed):** `GraniteMoeHybridTopKGating.forward` calls
   `expert_size.tolist()` on a device tensor at line 1046 of
   `modeling_granitemoehybrid.py`. This triggers a PJRT device-to-host transfer
   that fails on TT silicon with INTERNAL error code 13. The failure surfaces
   during `torch_xla.sync()` in `extract_compiled_graph`, i.e. when the first
   compiled sub-graph containing `shared_mlp` is dispatched to the device
   after the graph break caused by `expert_size.tolist()`.

   The companion `GraniteMoeHybridParallelExperts.forward` uses
   `inputs.split(expert_size, dim=0)` where `expert_size` is the `.tolist()`
   output — a Python list of per-expert counts with dynamic values — which
   also prevents static compilation.

2. **Compiler-stack precision issue (unfixed, Tier B):** After the loader fix,
   the test compiles and executes (64 minutes on Blackhole) but produces
   PCC=0.907 vs the required 0.99. The precision degradation comes from the
   Mamba SSM slow path across 40 layers: `segment_sum` expands `A` to a
   `[bsz, num_heads, num_chunks, chunk_size, chunk_size]` tensor (up to
   `[1, 48, 1, 256, 256]`), applies `torch.cumsum` on BF16 data, then masks
   with `-inf` and computes `torch.exp`. With 40 Mamba layers accumulating
   error through these operations, the output drifts to PCC≈0.907.  Whether
   this is a TT compiler math-fidelity setting or an inherent BF16 floor for
   this architecture requires precision analysis (compare BF16-CPU vs TT for
   the Mamba path alone) — that diagnosis precedes any fix.

## Fix
**Loader fix (committed):** Monkey-patched two classes at model load time:

- `_patched_topk_gating_forward`: Recomputes routing without calling
  `.tolist()`. Returns `sorted_expert_ids` as an `int32` tensor (replacing
  the Python list `expert_size`).

- `_patched_parallel_experts_forward`: Replaces `inputs.split(expert_size)`
  with a static for-loop over experts (`for e in range(self.num_experts)`).
  For each expert `e`, computes `F.linear(inputs, self.weight[e])` and masks
  out tokens not assigned to `e` via a boolean comparison. All operations
  remain in tensor-land with no Python-level splits or device→host transfers.

File changed: `tt_forge_models/granitemoehybrid/causal_lm/pytorch/loader.py`

**Proposed fix for PCC (Tier B):** Investigate whether the Mamba SSM precision
floor is caused by insufficient math fidelity in the TT compiler's `ComputeConfig`
(similar to the pattern seen in Gemma 7B, tracked in tt-xla #2861), or by
inherent BF16 accumulation. If compiler-caused, enable `fp32_dest_acc_en` or
raise math fidelity for the `cumsum`/`exp` kernels. If BF16 floor, add
`assert_pcc: false` to the test config with documented measured floor value.

## Tier B justification
cross-cutting — PCC degradation spans all 40 Mamba layers (segment_sum,
cumsum, exp), requiring investigation of math fidelity settings across multiple
kernels. Additionally the root cause (compiler bug vs BF16 floor) is unknown
and requires precision analysis before a targeted fix is possible.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    3857.39s (1:04:17) — with loader fix applied
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/granitemoehybrid/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4823272942056efab56a523b1f26232ec1f157c3 |
| tt-forge-models | 8872f513ac217967e88920d11ab89c1249a799b0 |
