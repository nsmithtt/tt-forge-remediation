# Remediation Summary: ms_lc_eq_d_vr_vae-pytorch-b4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ms_lc_eq_d_vr_vae/pytorch-b4-single_device-inference]

## Result
FAIL — PCC=0.839 after loader fix; Tier B BF16 matmul precision floor in full VAE encoder+decoder pass

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (at main HEAD commit 0f7b734348):
```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```
(Presented in CI as a Python crash dump: "Extension modules: numpy._core._multiarray_umath, ... (total: 222)")

After applying hf-bringup-38 loader fixes:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8390468601442368. Required: pcc=0.99.
```

## Root cause

**Loader bug (original crash):** At main HEAD (0f7b734348), `ModelLoader.load_inputs` had
signature `def load_inputs(self, **kwargs)` with `dtype_override` extracted via
`kwargs.get("dtype_override", torch.float32)`. `TorchDynamicLoader.load_inputs` uses
`inspect.signature` to detect the `dtype_override` parameter — if it is not a named
parameter, no bfloat16 override is passed. Since `dtype_override` was in `**kwargs`
only, `TorchDynamicLoader` fell through to `loader.load_inputs()` without any kwarg,
returning float32 inputs. Meanwhile `load_model` (explicit `dtype_override`) loaded
the model as bfloat16. The float32/bfloat16 mismatch caused a `RuntimeError` in
`conv_in` which propagated to a crash dump from the TT runtime.

Additionally, the default `vae_type` was `"decoder"`, supplying `(1, 4, 8, 8)` latent
inputs to a forward path that always calls `self.encode(x)` (expecting 3-channel image
input). This caused a channel-count mismatch.

The hf-bringup-38 branch (commit db6fc0722f and predecessors) fixed both issues:
1. Added explicit `dtype_override` keyword-only parameter to `load_inputs` so
   `TorchDynamicLoader` can detect and pass bfloat16.
2. Changed default `vae_type` from `"decoder"` to `"encoder"`, providing correct
   `(1, 3, 64, 64)` image-space inputs.

**Residual PCC failure (Tier B):** After the loader fix, the full `AutoencoderKL`
forward pass (encode + sample.mode() + decode) runs on TT silicon and produces
PCC=0.839 vs the CPU bfloat16 reference. The model has 60+ Conv2d layers and
GroupNorm operations in encoder + decoder (128→256→512 channels, 4 down-blocks +
mid attention + 4 up-blocks). BF16 accumulation error compounds through these
many layers, landing in the known `ttmlir-bf16-matmul-precision-floor` range.
The fix requires either f32 accumulation preservation through tt-mlir lowering
passes or per-op upcasting — both are cross-cutting changes.

## Fix
The loader fix is on the hf-bringup-38 branch
(commit db6fc0722f in tt-forge-models and earlier commits that added the explicit
`dtype_override` parameter). No new commits were produced in this remediation session.

For the PCC issue: the fix would live in tt-mlir's StableHLO→TTIR lowering passes,
preserving f32 accumulation in Conv2d and GroupNorm reduction paths. This is a
cross-cutting change across multiple lowering pass files.

## Tier B justification
cross-cutting — fixing the BF16 accumulation floor for Conv2d + GroupNorm requires
coordinated changes across multiple lowering passes in tt-mlir (or a model-level
precision override infrastructure), with risk of regressions in unrelated tests.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    144.73s (2:24)
- Tier A attempts: N/A

## Files changed
- `ms_lc_eq_d_vr_vae/pytorch/loader.py` (in tt-forge-models, on hf-bringup-38 branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8d262eddc30c019b006e1b7a743e0afa466ff5be |
