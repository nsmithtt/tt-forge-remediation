# Remediation Summary: qinglong_detailed_eyes_z_image-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qinglong_detailed_eyes_z_image/pytorch-Base-single_device-inference]

## Result
FAIL — complex tensor legalization fails in ZImageTransformer2DModel RoPE (xla-complex-tensor-not-supported)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
xla-complex-tensor-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original loader failure:
```
AssertionError
tests/infra/testers/single_chip/model/torch_model_tester.py:113: assert isinstance(self._model, torch.nn.Module)
```

After loader fix, compiler failure:
```
loc("p0.1"): error: failed to legalize unresolved materialization from ('tensor<1536x16x2xf32>') to ('tensor<1536x16xcomplex<f32>>') that remained live after conversion
ValueError: Error code: 13
```

## Root cause
Two bugs were found:

**Bug 1 (loader, fixed):** The `load_model()` method returned the full `DiffusionPipeline` instead of `pipeline.transformer`. `DiffusionPipeline` is not a `torch.nn.Module`, so the test framework assertion `isinstance(self._model, torch.nn.Module)` failed. Additionally, `load_inputs()` returned a dict with a plain text prompt instead of the latent tensors, timestep, and prompt embeddings that `ZImageTransformer2DModel.forward()` expects.

**Bug 2 (tt-mlir, unfixed, Tier B):** After the loader fix, the model compiles partially but the `_prepare_sequence` method (which implements RoPE for the ZImage transformer) uses complex tensors via `view_as_complex`. During StableHLO→TTIR lowering in tt-mlir, the complex tensor type `tensor<1536x16xcomplex<f32>>` cannot be materialized from the real representation `tensor<1536x16x2xf32>`. This results in an unresolved materialization error and an INTERNAL:13 error from PJRT.

This is the same root cause as the Lumina2 complex RoPE failure (`xla-complex-tensor-not-supported`).

## Fix
**Bug 1 (loader, applied):** In `qinglong_detailed_eyes_z_image/pytorch/loader.py`:
- Changed `load_model()` to call `pipeline.fuse_lora()` and return `pipeline.transformer` instead of the whole pipeline
- Changed `load_inputs()` to call `pipeline.encode_prompt()` and construct proper latent/timestep/prompt_embeds tensors matching `ZImageTransformer2DModel.forward(x, t, cap_feats)` signature
- Changed default dtype from `float16` to `bfloat16` (consistent with other Z-Image loaders)

**Bug 2 (compiler, not fixed):** Would require implementing complex tensor support (or a `view_as_complex` → real-valued decomposition pass) in the StableHLO→TTIR lowering pipeline in tt-mlir. This is new-infrastructure work.

## Tier B justification
Indicator: **new-infrastructure** — supporting complex float tensors requires new lowering patterns throughout the StableHLO→TTIR pipeline in tt-mlir. The `view_as_complex` operation (which reinterprets pairs of real values as complex numbers) has no existing lowering in the TT compiler stack. A fix would require either: (a) implementing complex type support end-to-end in tt-mlir, or (b) adding a pass that decomposes complex tensor ops back into real operations before any lowering occurs.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    184.54s (0:03:04)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/qinglong_detailed_eyes_z_image/pytorch/loader.py` (loader fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6c9311431a86568deac3a79c98cb0d31b6826e91 |
