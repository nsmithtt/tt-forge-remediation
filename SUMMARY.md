# Remediation Summary: isaac-pytorch-0.2_2B_Preview-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[isaac/pytorch-0.2_2B_Preview-single_device-inference]

## Result
FAIL — SpeculationLogDivergence in Dynamo recompilation after TensorStream graph break, caused by minicpm loaders' global nn.Module.__getattr__ patch contamination

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
dynamo-speculation-log-divergence-from-global-module-getattr-patch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
torch._dynamo.symbolic_convert.SpeculationLogDivergence:
SpeculationLog diverged at index 0 (log had 79 entries):
- Expected: /home/nsmith/tt-forge-remediation/tt-xla/third_party/tt_forge_models/minicpmv_2_6/pytorch/loader.py:46 (CALL at ip=19)
- Actual: /home/nsmith/tt-forge-remediation/tt-xla/third_party/tt_forge_models/minicpm_o_2_6/pytorch/loader.py:48 (CALL at ip=19)

from user code:
   File "modular_isaac.py", line 2178, in torch_dynamo_resume_in_forward_at_2173
    outputs = self.model(
  File "minicpm_o_2_6/pytorch/loader.py", line 48, in patched_getattr
    return original_getattr(self, name)
```

## Root cause

Six separate loader-layer bugs were fixed before reaching this terminal failure:

**Bug 1 (loader):** `SlidingWindowCache` removed from `transformers.cache_utils` in 5.x. Fixed: inject a stub subclass of `Cache`.

**Bug 2 (loader):** `DefaultFastImageProcessorKwargs` renamed to `ImagesKwargs` in `transformers.image_processing_utils_fast`. Fixed: alias back.

**Bug 3 (loader):** `TensorType` moved from `transformers.tokenization_utils` to `tokenization_utils_base`. Fixed: inject back.

**Bug 4 (loader):** `rope_theta` must live inside `rope_parameters` dict (transformers 5.x). `IsaacConfig.__init__` overwrites `rope_parameters` with an incomplete `_rope_scaling` dict (no `rope_theta`). Fixed: patch `Qwen3RotaryEmbedding` and `Qwen2_5_VLRotaryEmbedding.compute_default_rope_parameters` to fall back to `config.rope_theta`.

**Bug 5 (loader):** Isaac's chat template only handles string content; the original loader passed list-format `[{"type": "image"}, ...]` content which produces an empty string (causing `"Number of <image> tokens in text (0) must match number of images (1)"`). Fixed: use string content `"<image>\nWhat is shown in this image?"` and pass `images=[image]`.

**Bug 6 (loader):** `"eager"` key absent from `ALL_ATTENTION_FUNCTIONS` in transformers 5.x (only `"paged|eager"` exists). `modular_isaac.py` captures `_ORIGINAL_ATTENTION_FUNCTIONS["eager"]` at import time; if absent, `_isaac_eager_forward` raises `ValueError("Base eager attention function unavailable for fallback.")` even for `IsaacVisionAttention` modules. Fixed: alias `"eager"` → `"paged|eager"` before the HF dynamic module is imported.

**Bug 7 (loader):** `attn_implementation="eager"` in `from_pretrained` propagates recursively to `vision_config._attn_implementation`, overriding the pre-load `"sdpa"` setting. `_isaac_eager_forward`'s packed-sequence matmul is `[L,H,D]@[L,D,H]=[L,H,H]` (wrong; should be `[H,L,L]`), producing a shape mismatch with the block-diagonal mask `[L,L]`. Fixed: after `from_pretrained`, iterate `model.modules()` and set `vision_config._attn_implementation = "sdpa"` on the first `IsaacVisionAttention` module (all share the same config object).

**Bug 8 (tt-xla, Tier A fixed):** `prims::view_of` is a non-ATen op with alias annotations. `partition_fx_graph_for_cpu_fallback` cannot functionalize it. Fixed: cherry-pick `bypass_prims_view_of` pass from `remediation/isaac_0_1-pytorch-Isaac_0_1-single_device-inference` branch into `python_package/tt_torch/backend/passes.py` and `backend.py`.

**Terminal failure (Tier B):** After all loader fixes and the `prims::view_of` fix are applied, the model reaches Dynamo compilation. `IsaacForConditionalGeneration.forward` calls `modality_mask(tensor_stream)` where `TensorStream` is an opaque custom type from `perceptron.tensorstream.ops` that Dynamo cannot trace. This causes a graph break; Dynamo resumes from `torch_dynamo_resume_in_forward_at_2173` for `outputs = self.model(...)`.

The `self.model` attribute access goes through `nn.Module.__getattr__`, which is globally patched by both `minicpm_o_2_6` and `minicpmv_2_6` loaders at module import time. Both loaders install `patched_getattr` functions that chain together. The chain order (and thus which function appears at the top) differs between the first Dynamo compilation pass (which builds the 79-entry speculation log) and the second pass triggered by the graph break. This causes `SpeculationLogDivergence` at index 0.

The CPU-only forward pass (without torch.compile) succeeds correctly.

## Fix
Seven loader bugs fixed in `tt-forge-models/isaac/pytorch/loader.py` (remediation branch `remediation/isaac-pytorch-0.2_2B_Preview-single_device-inference`).

One Tier A compiler fix cherry-picked into `tt-xla/python_package/tt_torch/backend/passes.py` and `backend.py`: `bypass_prims_view_of` pass replaces `prims.view_of(x)` nodes with their input `x` in the FX graph before XLA bridge.

**Proposed fix for terminal failure:** The `minicpm_o_2_6` and `minicpmv_2_6` loaders both patch `nn.Module.__getattr__` at module-import time. These patches must be made idempotent and stable (e.g., by checking if the patch is already applied before overwriting, or by using a single shared wrapper that both loaders register into). Additionally, the test framework's `run_around_tests` fixture should call `torch._dynamo.reset()` BEFORE the test (not just after) to prevent stale speculation logs from previous compilations from being used in the current test's recompilation.

## Tier B justification
`cross-cutting` — fixing the `SpeculationLogDivergence` requires changes in at least three places:
1. `minicpm_o_2_6/pytorch/loader.py` — stabilize the `__getattr__` patch chain
2. `minicpmv_2_6/pytorch/loader.py` — same stabilization
3. `tests/conftest.py` (tt-xla) — reset Dynamo before each test, not just after

The root cause (TensorStream as an opaque type causing graph breaks in Isaac's model) is inherent to the model architecture and cannot be fixed from within the loader without CPU-offloading the vision preprocessing (forbidden).

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    64.05s (1:04) to SpeculationLogDivergence
- Tier A attempts: 1 (bypass_prims_view_of cherry-pick — correctly applied, unblocked prims::view_of error)

## Files changed
- `tt-forge-models/isaac/pytorch/loader.py` — six transformers 5.x compat fixes + vision attention sdpa override
- `tt-xla/python_package/tt_torch/backend/passes.py` — `bypass_prims_view_of` function
- `tt-xla/python_package/tt_torch/backend/backend.py` — wire `bypass_prims_view_of` into `torch_pass_pipeline`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | (via tt-xla submodule, same configured env) |
| tt-xla          | f399962217f1ee61877bea7ad42b40e96a160e20 |
| tt-forge-models | 91bcc49d6d29a441d8b42e10be6dfdfbb1db65da |
