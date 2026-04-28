# Remediation Summary: phi3-phi_3_5-pytorch-phi_3_5_moe_tiny_random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[phi3/phi_3_5/pytorch-Phi_3.5_Moe_Tiny_Random-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
phimoe-experts-eager-dynamic-loop-index-add

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Raised during `torch_xla.sync()` inside `extract_graph_helper` while compiling the
`PhimoeExperts` forward subgraph. A prior TT_FATAL also appeared:

    TT_FATAL: Number of rows in gradient tensor must be equal to number of indices in index tensor
    (embedding_backward_device_operation.cpp:67:
     grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1])

## Root cause
Two loader bugs combined to produce the failure.

**Bug 1 — trust_remote_code loading wrong class (commit bf3c99f56b):**
`trust_remote_code=True` caused `AutoModelForCausalLM.from_pretrained` to load the
custom `modeling_phimoe.py` from the checkpoint repo. That custom class expects weight
keys in a different format (`block_sparse_moe.experts.{i}.w1/w2/w3`) than the native
transformers `PhimoeForCausalLM` (which uses batched `gate_up_proj`/`down_proj`). All
expert weights loaded as randomly initialized, and the model produced garbage output.

**Bug 2 — eager MoE expert loop creates dynamic shapes (commit 42b883c6bd):**
The "eager" `PhimoeExperts.forward` implementation (selected when
`_experts_implementation` is `None` or `"eager"`) uses:
1. `expert_hit = torch.greater(...).nonzero()` — tensor with a data-dependent number of rows
2. `for expert_idx in expert_hit:` — Python loop over a dynamic-length tensor
3. `final_hidden_states.index_add_(0, token_idx, ...)` — scatter-add with dynamic indices

`nonzero()` causes a graph break, splitting the MoE block into multiple XLA subgraphs.
When XLA compiles the subgraph containing `index_add_`, it lowers the operation to
`ttnn::embedding_backward`. The TTNN embedding backward kernel asserts
`grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]`, which fails
because the dynamic shapes produced by `nonzero()` do not satisfy the expected static
relationship. This propagates as `INTERNAL: Error code: 13` when `torch_xla.sync()` is
called.

## Fix
**Fix 1 (commit bf3c99f56b):** Remove `trust_remote_code=True` from both `_load_tokenizer`
and `load_model` for the `TINY_RANDOM_MOE` variant. The transformers 5.x native
`PhimoeForCausalLM` loads correctly without it and uses the correct weight layout.

**Fix 2 (commit 42b883c6bd):** After loading, set
`model.config._experts_implementation = "batched_mm"`. The `batched_mm` implementation
(`transformers/integrations/moe.py:batched_mm_experts_forward`) is fully vectorized:
- Uses `torch.arange` + `reshape` (static shapes, no `nonzero()`)
- Uses `view(num_tokens, num_top_k, hidden_dim).sum(dim=1)` instead of `index_add_`
- No Python-level control flow on tensor values, no graph breaks

Both fixes are in `tt-xla/third_party/tt_forge_models/phi3/phi_3_5/pytorch/loader.py`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    29.00s (36.49s first run)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/phi3/phi_3_5/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e8df5dfa88f8726cd6b965996f7c1af649187a77 |
| tt-forge-models | 42b883c6bd4e64865d4b49b1672f65851deb0365 |
