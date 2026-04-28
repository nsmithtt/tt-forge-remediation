# Remediation Summary: dasiwa-pytorch-DASIWA-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dasiwa/pytorch-DASIWA-single_device-inference]

## Result
FAIL — SIGFPE during TT silicon execution of second compiled graph; `prior_embedding[prior_token_drop] *= 0.0` triggers `aten.nonzero.default` (dynamic output shape), which crashes the TT backend with a floating-point exception

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
aten-nonzero-dynamic-output-shape-sigfpe

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original reported failure was:
```
The image processor of type `GlmImageImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```

When reproduced locally the first failure encountered was:
```
AttributeError: 'GlmImagePipeline' object has no attribute 'load_lora_weights'
```

After fixing the loader, the test fails with:
```
Fatal Python error: Floating point exception
  File "diffusers/models/transformers/transformer_glm_image.py", line 625 in torch_dynamo_resume_in_forward_at_625
  File "diffusers/models/transformers/transformer_glm_image.py", line 625 in forward
```

Line 625 is: `prior_embedding[prior_token_drop] *= 0.0`

## Root cause

Two bugs were found in sequence:

**Bug 1 (loader — fixed):** `GlmImagePipeline` does not inherit `LoraLoaderMixin`, so
`load_lora_weights` is absent. The original loader called it unconditionally,
raising `AttributeError`. Fixed by guarding with `hasattr` and extracting
only the transformer component with correct dummy inputs (4D `hidden_states`,
`bool` `prior_token_drop`).

**Bug 2 (compiler-stack — unfixed, Tier B):** The transformer forward at line 625
performs a boolean masked in-place assignment:
```python
prior_embedding[prior_token_drop] *= 0.0
```
`torch._dynamo.explain()` confirms this causes a graph break due to
`aten.nonzero.default` whose output shape depends on data (dynamic). torch.compile
creates a `torch_dynamo_resume_in_forward_at_625` fragment starting at this line.
When TT silicon executes this compiled fragment, a SIGFPE terminates the process.
The crash occurs immediately after `PJRT_Executable_GetCompiledMemoryStats` (stub),
consistent with execution of the second compiled graph on TT device.

The `aten.nonzero` operation with a dynamic-output-shape tensor is the immediate
trigger; TT silicon does not handle this correctly and crashes rather than raising
a Python exception.

## Fix

**Loader fix (committed):**
- `dasiwa/pytorch/loader.py` in `tt_forge_models` on
  `remediation/dasiwa-pytorch-DASIWA-single_device-inference`:
  - Guarded `load_lora_weights` with `hasattr(pipe, "load_lora_weights")`
  - Returns `pipe.transformer` (the `GlmImageTransformer2DModel`) instead of the full pipeline
  - Replaced text-prompt `load_inputs` with correct tensor inputs:
    4D `hidden_states (B, 16, 4, 4)`, `encoder_hidden_states (B, 8, 1472)`,
    `prior_token_id (B,)`, `prior_token_drop (B, bool)`, `timestep (B,)`,
    `target_size (B, 2)`, `crop_coords (B, 2)`

**Proposed compiler fix (not attempted — Tier B):**
The `aten.nonzero.default` dynamic-shape op needs either:
1. A proper TT kernel that supports `nonzero` (scatter-based index computation), or
2. The TT backend should detect this op and fall back to CPU eagerly instead of
   crashing with SIGFPE.

The fix would live in `tt-xla` (PJRT bridge — add eager fallback for dynamic-shape
ops) or `tt-mlir` (add lowering pattern for `aten.nonzero`). Both paths touch more
than one file and require new infrastructure.

## Tier B justification
Indicator: **internal-error-unknown-mechanism** + **new-infrastructure**

The SIGFPE is an unhandled crash from C code in the TT runtime with no error
message, making the exact lowering path that crashes unclear. Properly supporting
`aten.nonzero` (dynamic output shape) on TT silicon is new infrastructure: it
requires either a scatter-based kernel or a device→host fallback path, both
cross-cutting changes to the PJRT/XLA bridge.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    ~90s to SIGFPE after loader fix
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/dasiwa/pytorch/loader.py` — loader rewrite (guard LoRA loading,
  return transformer component, correct dummy inputs)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d7bf16a403be4d2822f64b9fd109bd777472e185 |
| tt-forge-models | 40577a2fe5e94dd1ebd33f4a34c25276b991dbb8 |
