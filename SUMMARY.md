# Remediation Summary: kimi_k2_instruct-pytorch-baseten-FP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kimi_k2_instruct/pytorch-baseten-FP4-single_device-inference]

## Result
FAIL — Tier B BF16 precision in noaux_tc MoE routing causes PCC=0.81 (required 0.99) after loader fix

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-moe-routing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):

```
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
call_function <Wrapped method <original add>>(*(FakeTensor(..., size=(), dtype=torch.int64), 0), **{}):
got AttributeError("'ndarray' object has no attribute 'add'")

from user code:
   File "modeling_deepseek.py", line 580, in moe_infer
    end_idx = start_idx + num_tokens
```

Residual failure (after loader fix):

```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.8090801803466806. Required: pcc=0.99.
```

## Root cause

**Loader bug (fixed):** `DeepseekV3MoE.moe_infer` calls
`tokens_per_expert.cpu().numpy()` then iterates: `end_idx = start_idx + num_tokens`.
Under Dynamo's `TorchFunctionMode`, integer arithmetic on numpy scalars is wrapped
through `__torch_function__`, which calls `.add()` on the ndarray — a method that
doesn't exist. Fix: replace `.cpu().numpy()` with `.cpu().tolist()` so loop variables
are Python ints (Dynamo treats them as scalar constants, not traced tensors).

**Residual Tier B bug:** The `noaux_tc` routing algorithm uses two layers of discrete
`topk` selection. With 256 experts divided into `n_group=8` groups and `topk_group=4`,
the gate first computes per-group scores via `.topk(2, dim=-1)[0].sum(dim=-1)` and
then selects the top-4 groups. With random model weights the gate scores across the
256 experts are nearly uniform (≈ 1/256 each), so any BF16 accumulation difference
between TT's matmul implementation and CPU's flips which groups — and therefore which
experts — are selected. Because the expert outputs are entirely determined by which
experts are routed to, flipping the group selection produces completely different
logits. PCC=0.81 (rather than near 0) is consistent with partial group overlap (some
of the 4 selected groups match, some do not). This is a fundamental sensitivity of
`noaux_tc` MoE routing to BF16 precision with random weights.

## Fix

**Loader fix (committed):** Monkey-patch `DeepseekV3MoE.moe_infer` in
`kimi_k2_instruct/pytorch/loader.py` to replace every `.cpu().numpy()` call with
`.cpu().tolist()`, ensuring the loop-body integer arithmetic uses Python ints rather
than numpy scalars. The patch function `_fixed_moe_infer` is installed via
`sys.modules` scan for `"modeling_deepseek"` after `get_class_from_dynamic_module`
completes.

Committed on `remediation/kimi_k2_instruct-pytorch-baseten-FP4-single_device-inference`
in `tenstorrent/tt-forge-models`.

**Proposed fix for Tier B PCC bug:** TT-MLIR's BF16 matmul accumulation precision
would need to exactly match CPU's for models using discrete topk routing, OR the
model configuration would need to use a routing algorithm that is robust to small
score differences (e.g., deterministic tiebreaking or higher-precision gate
projection). This is a cross-cutting change in `tt-mlir` affecting how BF16 matmul
accumulation is rounded.

## Tier B justification

Indicator: **cross-cutting** — making TT's BF16 matmul accumulation match CPU exactly
would require coordinated changes across multiple MLIR lowering passes (matmul →
TTIR → TTMetal kernel configuration). Even then, the fix would affect all BF16 models,
not just MoE, making it a cross-cutting change. Alternatively, fixing the routing
sensitivity requires model-level changes outside the compiler stack scope.

## Verification
- pytest exit: FAIL (PCC=0.809 < 0.99) — after loader fix resolves TorchRuntimeError
- Hardware:    wormhole
- Duration:    131.02s (0:02:11)
- Tier A attempts: N/A

## Files changed
- `kimi_k2_instruct/pytorch/loader.py` in `tenstorrent/tt-forge-models`
  — added `_fixed_moe_infer` patch function and monkey-patch call in `load_model()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3f03af5df2a26bf0d6a71bd125789c59114d6f70 |
