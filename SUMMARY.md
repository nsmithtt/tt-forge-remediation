# Remediation Summary: ivao0_voc-pytorch-voc-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ivao0_voc/pytorch-voc-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-all-tied-weights-keys-missing-post-init

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'Voc' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?

(followed by RuntimeError: expected scalar type Float but found BFloat16, then
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node: aten.add_.Tensor found two different devices xla:0, cpu)

## Root cause

Three loader-layer bugs in `ivao0_voc/pytorch`:

1. **Missing `post_init()` in `Voc.__init__`** — transformers 5.x `_finalize_model_loading` calls `model._adjust_tied_keys_with_tied_pointers` which accesses `self.all_tied_weights_keys`. This attribute is initialized by `post_init()`. `Voc.__init__` called `super().__init__(config)` but never called `self.post_init()`, so `all_tied_weights_keys` was absent. Same pattern as H2OVL / InternVL3.

2. **`acoustic` tensor dtype mismatch in `SplitResidualVectorQuantizer.decode`** — `acoustic` was initialized with `torch.zeros([1, 1])` which produces float32. When the model is loaded with `dtype_override=torch.bfloat16`, the codebook lookup results are BF16, but `acoustic` stays float32 through the accumulation loop. The final `self.out_proj_a(acoustic)` then gets a float32 input against a BF16 weight, raising a dtype mismatch.

3. **Streaming state cross-device conflict in `BufferConvTranspose1d` / `BufferConv1d`** — Both conv classes store per-call state (`self.partial`, `self.previous`) as plain Python attributes (not registered buffers). The CPU inference run sets these to CPU tensors. When TorchDynamo traces the model for XLA compilation, it encounters `self.partial` (a CPU tensor) being added to `out` (an `xla:0` FakeTensor), triggering "two different devices xla:0, cpu". Fix: reset all streaming state to `None` at the start of each `VocDecodeWrapper.forward` so both CPU and XLA runs start from clean state.

## Fix

All three fixes are in `third_party/tt_forge_models/ivao0_voc/pytorch/`:

1. `src/model.py` — added `self.post_init()` at the end of `Voc.__init__`.
2. `src/model.py` — changed `torch.zeros([1, 1], device=codes.device)` to `torch.zeros([1, 1], device=codes.device, dtype=semantic.dtype)` in `SplitResidualVectorQuantizer.decode`.
3. `loader.py` — imported `BufferConv1d, BufferConvTranspose1d` from `src.model`; added state-reset loop at the top of `VocDecodeWrapper.forward` that sets `m.previous = None` / `m.partial = None` for all matching sub-modules before each forward call.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    114.94s (0:01:54)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/ivao0_voc/pytorch/src/model.py`
- `third_party/tt_forge_models/ivao0_voc/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4b873cc7163f396b52df64e30774e278401623fa |
| tt-forge-models | 04f77c52778910a66d96a86b144ba0582690e554 |
