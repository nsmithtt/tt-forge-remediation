# Remediation Summary: convit-pytorch-Base_FB_IN1K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[convit/pytorch-Base_FB_IN1K-single_device-inference]

## Result
SILICON_PASS — all three loader/infra bugs fixed; test passes on TT silicon

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
convit-rel-indices-not-registered-buffer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake
tensors: call_module L__self___blocks_0_attn_pos_proj(*(FakeTensor(..., size=(1,
196, 196, 3), dtype=torch.bfloat16),), **{}): got RuntimeError('Unhandled
FakeTensor Device Propagation for aten.mm.default, found two different devices
cpu, xla:0')

Preceded by two import-time blockers that had to be fixed first:
1. NameError: name 'Mesh' is not defined  (torch_multichip_utils.py)
2. AttributeError: module 'spacy' has no attribute 'Language'  (datasets/dill)

## Root cause

Three stacked loader/infra bugs, each of which had to be fixed in sequence:

**Bug 1 — torch_multichip_utils.py NameError (tt-xla infra)**
Commit 0afacef3 wrapped `torch_xla` imports in `try/except ImportError` to
support environments without torch_xla, but left the `-> Mesh` return-type
annotation on `get_mesh()`. Python 3.12 evaluates annotations eagerly at
function-definition time, so when `torch_xla` is absent `Mesh` is undefined
and importing the module raises `NameError`, blocking all pytest collection.
Fix: add `from __future__ import annotations` to defer annotation evaluation.

**Bug 2 — huspacy loader top-level `import spacy` (tt_forge_models)**
`setup_models_path` adds the `tt_forge_models` root directory to `sys.path`
so that relative imports inside loader modules work. `tt_forge_models/spacy/`
exists (without an `__init__.py`), so Python 3 treats it as a namespace
package. When the huspacy loader is collected during pytest, its module-level
`import spacy` resolves to this empty namespace package and stores it in
`sys.modules['spacy']`. Later, the `datasets` library's `dill` serializer
checks `if "spacy" in sys.modules:` and then tries `spacy.Language`, which
fails with `AttributeError` because the namespace package has no `Language`
attribute. Fix: move `import spacy` from module level into `_load_nlp()`.

**Bug 3 — ConViT `rel_indices` not a registered buffer (tt_forge_models)**
`timm`'s `MHSA_rotary_pos` initializes `self.rel_indices` as a plain tensor
attribute (not via `register_buffer`). The first forward pass triggers lazy
computation and stores the result (shape 1×196×196×3, CPU) in `__dict__`.
Because it is not a registered buffer, `model.to(xla_device)` — called by the
test runner before torch.compile — moves all parameters to XLA but leaves
`rel_indices` on CPU. During dynamo FakeTensor tracing:
  - `pos_proj.weight` is a FakeTensor on `xla:0` (it moved with the model)
  - `pos_score = self.rel_indices.expand(B, …)` is a FakeTensor on `cpu`
    (lifted from the real CPU tensor in `__dict__`)
  - `aten.mm.default(pos_score_cpu, weight_xla)` → device mismatch error

## Fix

**Fix 1** — `tt-xla/tests/infra/utilities/torch_multichip_utils.py`
Add `from __future__ import annotations` at the top of the file so that the
`-> Mesh` annotation is not evaluated at import time.

**Fix 2** — `tt_forge_models/huspacy/pytorch/loader.py`
Remove `import spacy` from module level; add it inside `_load_nlp()` so it
is only imported when the HuSpaCy model is actually loaded.

**Fix 3** — `tt_forge_models/convit/pytorch/loader.py`
In `load_model()`: run a dummy forward pass (`torch.zeros(1, 3, 224, 224)`)
to trigger `get_rel_indices` for the 196-patch configuration, then iterate
through all model modules and, for any that have a `rel_indices` tensor
attribute not already registered as a buffer, call `register_buffer` with a
detached clone. After this, `model.to(device)` moves `rel_indices` to the
correct device alongside all other parameters and buffers.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    61.98s
- Tier A attempts: N/A

## Files changed
- tt-xla: `tests/infra/utilities/torch_multichip_utils.py`
- tt_forge_models: `huspacy/pytorch/loader.py`
- tt_forge_models: `convit/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0825039587ab3d19ca30457795e04787f5348017 |
| tt-forge-models | 004f19a4cf30e507a2a1701bd1719ad28d7310d7 |
