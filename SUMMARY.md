# Remediation Summary: kimi_k2_instruct_quantized_w4a16-causal_lm-pytorch-Kimi-K2-Instruct-quantized.w4a16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2_instruct_quantized_w4a16/causal_lm/pytorch-Kimi-K2-Instruct-quantized.w4a16-single_device-inference]

## Result
FAIL — static moe_infer unrolls 384 experts into a graph too large to compile on TT silicon

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
moe-static-unroll-compilation-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}):
got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File "modeling_deepseek.py", line 580, in moe_infer
    end_idx = start_idx + num_tokens
```

## Root cause
Three loader-layer bugs prevented the test from running, plus a subsequent compiler-stack bug:

**Bug 1 (loader):** The model's remote `tokenization_kimi.py` imports `bytes_to_unicode` from
`transformers.models.gpt2.tokenization_gpt2`, which was removed in transformers 5.x.

**Bug 2 (loader):** `DynamicCache.get_usable_length` was removed in transformers 5.x; the model's
remote `modeling_deepseek.py` calls it during the model forward pass.

**Bug 3 (loader):** `DeepseekV3MoE.moe_infer` calls `tokens_per_expert.cpu().numpy()` to obtain
expert routing counts, then uses the numpy array in an arithmetic loop. During Dynamo tracing,
`cpu().numpy()` on a FakeTensor returns a numpy array wrapping FakeTensors; on the second loop
iteration `start_idx` has become a FakeTensor and `FakeTensor + numpy_scalar` raises
`AttributeError("'ndarray' object has no attribute 'add'")`.

The fix for Bug 3 replaces `moe_infer` with a static per-expert masked matmul (looping over all
`len(self.experts)` = 384 experts with gate weights). This eliminates the D2H dependency and the
Dynamo error. However, Dynamo unrolls the Python `for e in range(384)` loop into 384 × 3 linear
ops in the FX graph, creating an enormous graph that tt-mlir cannot compile in reasonable time
(tested: >28 minutes, killed).

**Tier B** (moe-static-unroll-compilation-hang): Efficiently handling 384-expert static MoE
unrolling requires either batched-matmul infrastructure in tt-mlir (so that 384 identical linear
ops are fused/batched) or a different dispatch mechanism (e.g. `_experts_implementation="batched_mm"`
in the loader that uses `torch.bmm` with pre-stacked weights). Either option is cross-cutting and
requires more than a one-file fix.

## Fix
Three loader fixes were applied in
`kimi_k2_instruct_quantized_w4a16/causal_lm/pytorch/loader.py` on
`remediation/kimi_k2_instruct_quantized_w4a16-causal_lm-pytorch-Kimi-K2-Instruct-quantized.w4a16-single_device-inference`:

1. Inject `bytes_to_unicode` from `transformers.models.clvp.tokenization_clvp` into
   `transformers.models.gpt2.tokenization_gpt2` (removed in transformers 5.x).

2. Add `DynamicCache.get_usable_length` shim that delegates to `get_seq_length(layer_idx)`
   (method removed in transformers 5.x).

3. Patch `DeepseekV3MoE.moe_infer` via `get_class_from_dynamic_module` with a static per-expert
   masked matmul that builds `gate_matrix[n,e] = weight if token n routes to expert e` and
   accumulates `sum_e(expert_e(x) * gate_matrix[:, e])` without any D2H transfer.

**Proposed fix for Tier B:** Replace the static `for e in range(num_experts)` loop with a
batched matmul using pre-stacked weight buffers:
```python
# Pre-stack once: [E*D_i, D_h] for gate/up, [E, D_h, D_i] for down
gate_proj_out = F.linear(x, self._gate_w_stacked).view(N, E, D_i)
up_proj_out   = F.linear(x, self._up_w_stacked).view(N, E, D_i)
hidden = act_fn(gate_proj_out) * up_proj_out              # [N, E, D_i]
expert_outs = torch.bmm(
    hidden.transpose(0,1),                                # [E, N, D_i]
    self._down_w_stacked.transpose(1,2)                   # [E, D_i, D_h]
).transpose(0,1)                                          # [N, E, D_h]
output = (expert_outs * gate_matrix.unsqueeze(-1)).sum(1) # [N, D_h]
```
This creates a 5-op graph (2 F.linear, 1 bmm, 2 element-wise) regardless of expert count, but
requires validating that tt-mlir handles the large matmul dims (E*D_i = 786432 output features)
and the 3-D bmm ([384, N, D_i] × [384, D_i, D_h]).

## Tier B justification
**Indicator:** cross-cutting — the compilation hang from unrolling 384 experts into
identical ops in the FX graph requires either FX-graph-level op fusion (tt-mlir pass) or a new
loader-level `batched_mm` dispatch validated end-to-end against tt-metal bmm with large batched
dimensions. This mirrors the `moe-static-unroll-dram-overflow` Tier B from DeepSeek V3 (256
experts, OOM on BH). Cannot be fixed by a single scoped change in one file.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150)
- Duration:    >28 min (compilation hang, killed)
- Tier A attempts: N/A

## Files changed
- `kimi_k2_instruct_quantized_w4a16/causal_lm/pytorch/loader.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d22f12342784c3a59f92dea556aa813403c69ff7 |
| tt-forge-models | bac2a693beaf638c9722005dcb7f9e3047b1bfbc |
