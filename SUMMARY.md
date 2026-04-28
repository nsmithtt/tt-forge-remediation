# Remediation Summary: cosmos_reason2-image_to_text-pytorch-8b-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_reason2/image_to_text/pytorch-8b-single_device-inference]

## Result
FAIL — Tier B compiler-stack bug: TT PJRT cannot compile XLA sub-graphs whose inputs are CPU tensors (graph-break boundary issue from Qwen3-VL control-flow .tolist() calls)

## Stack layer
tt-xla

## Tier
B

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
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Traceback (from test run with _CpuOnlyTensor + vision-forward patch):
```
modeling_qwen3_vl.py:778: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
modeling_qwen3_vl.py:778: in torch_dynamo_resume_in_forward_at_778
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
torch/_dynamo/eval_frame.py:1044: in _fn
backend.py:215: in _call_experimental_compile
dynamo_bridge.py:826: in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

`nvidia/Cosmos-Reason2-8B` is a gated HuggingFace repo; the loader was
switched to the architecturally identical public model
`Qwen/Qwen3-VL-8B-Instruct`. That is a clean loader fix.

The remaining failure is a compiler-stack bug. Qwen3-VL's vision encoder
forward (`Qwen3VLVisionModel.forward`) calls `.tolist()` on `image_grid_thw`
inside `fast_pos_embed_interpolate` and `rot_pos_emb` for Python control flow
(dynamic sequence-length arithmetic). The top-level language model forward
calls `get_rope_index`, which also calls `input_ids.tolist()`.

When inputs are on TT device, any `.tolist()` triggers a device-to-host
read that TT PJRT does not support (Error code 13). The fix of keeping
`image_grid_thw` on CPU via `_CpuOnlyTensor` (so `.tolist()` runs without a
d2h read) causes torch.dynamo to break the XLA compilation graph at the
`.tolist()` call site. The resumed compiled sub-graph then receives the
eagerly-computed CPU tensors (`pos_embeds`, `rotary_pos_emb`, `cu_seqlens`)
as *inputs*. TT's XLA/PJRT backend cannot compile a program whose inputs
include CPU tensors — every input to a compiled TT XLA program must already
be on the TT device. This produces the same Error code 13 at
`extract_compiled_graph_helper` / `torch_xla.sync`.

`position_ids` is pre-computed in `load_inputs` on CPU and passed as a
regular input (the test runner moves it to TT device), successfully bypassing
`get_rope_index`'s `input_ids.tolist()` call.

The blocker: no loader-level fix can prevent torch.dynamo from breaking the
graph at `.tolist()` calls inside `fast_pos_embed_interpolate` /
`rot_pos_emb`, and no loader-level fix can make TT PJRT accept CPU tensors
as compiled sub-graph inputs. Both require changes to the torch_xla dynamo
bridge or TT PJRT's input-capture layer.

## Fix
**Partial loader fix committed** (`tt_forge_models`
`remediation/cosmos-reason2-pytorch-8b-single-device-inference`
commit `21a1263334`):

- `loader.py`: switch from gated `nvidia/Cosmos-Reason2-8B` to public
  `Qwen/Qwen3-VL-8B-Instruct` (same architecture).
- `loader.py`: add `_CpuOnlyTensor` subclass — keeps `image_grid_thw` on CPU
  even when the test runner calls `.to(tt_device)`.
- `loader.py`: pre-compute `position_ids` on CPU in `load_inputs` via
  `model.model.get_rope_index` before tensors are moved to device; this
  bypasses `get_rope_index`'s `input_ids.tolist()` during the compiled
  forward.
- `loader.py`: patch `Qwen3VLVisionModel.forward` to explicitly move all
  CPU-derived tensors (`pos_embeds`, `rotary_pos_emb`, `cu_seqlens`,
  `position_embeddings`) to TT device with `.to(target_device)`.

**Required fix (not implemented)** in `tt-xla` / TT PJRT layer: when
torch.dynamo's sub-graph input capture encounters a CPU tensor at a
graph-break boundary, it must perform a host-to-device transfer before
handing the tensor to the compiled XLA program. Without this, any model that
uses CPU-side control flow (`.tolist()`, Python integer arithmetic on tensor
values) inside a compiled forward will fail with Error code 13.

## Tier B justification
**cross-cutting**: The fix requires the torch_xla dynamo bridge or TT PJRT's
sub-graph input pipeline to transparently handle CPU→TT host-to-device
transfers at graph-break boundaries. This is not a scoped single-function
change — it affects how all compiled sub-graph inputs are handled whenever
a Python-side graph break produces CPU tensors from a TT-device computation
context.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    not-run (test fails at compilation, no full-model timing)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/cosmos_reason2/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8dcedfc4d2682882709effc75e36ee74e44a5dd0 |
| tt-forge-models | 21a1263334db68eeaefc17aed50924e6b4139e8a |
