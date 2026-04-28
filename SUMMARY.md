# Remediation Summary: gemma3-multimodal-pytorch-leon-se-gemma-3-27b-it-qat-W4A16-G128-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3/multimodal/pytorch-leon-se/gemma-3-27b-it-qat-W4A16-G128-single_device-inference]

## Result
FAIL — compressed_tensors W4A16 quantized ops crash the XLA CPU fallback partition logic (fused_0 has no xla_args)

## Stack layer
tt-xla

  - `loader`         — bug was in tt_forge_models or test inputs
  - `tt-xla`         — bug in compiler frontend (PJRT, torch_xla bridge)
  - `tt-mlir`        — bug in compiler core (StableHLO→TTIR lowering)
  - `tt-metal`       — bug in backend runtime / kernels
  - `hardware-class` — model exceeds single-device capacity (XFAIL)
  - `n/a`            — NO_FIX_NEEDED (could not reproduce)

## Tier
B

  - `N/A` — loader fix, no fix needed, or hardware-class XFAIL
  - `A`   — compiler-stack fix attempted (succeeded → SILICON_PASS,
            ran out of attempts → FAIL with explanation)
  - `B`   — compiler-stack bug filed without attempting fix

## Bug fingerprint
compressed-tensors-w4a16-xla-cpu-fallback-crash

  Format: `<area>-<short-description>`. Use the same string verbatim
  whenever a later report hits the same bug — this is how the audit
  groups failures.

  Examples drawn from existing failures:
    sdpa-k-chunk-size-lt-32
    pjrt-device-to-host-transfer
    conv2d-reader-indices-cb-page-size
    stablehlo-round-nearest-even-no-lowering
    aten-slice-tensor-out-of-bounds-start
    avg-pool2d-ceil-mode-zero-output
    ttmlir-f32-precision-not-preserved
    transformers-5x-use-fast-default       (loader bug)
    gguf-load-checkpoint-model-to-load-kwarg (loader bug)
    n/a                                    (NO_FIX_NEEDED only)

## Workaround self-check
Verify each rule from the Forbidden workarounds section. Mark NO if the
fix did not use the technique; YES with a measured justification only
when explicitly permitted (currently only PCC lowering for measured bf16
accumulation).

- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

After fixing the above loader bug (two commits on remediation branch), the test progresses to execution and hits:

AttributeError: 'fused_0' object has no attribute 'xla_args'

in venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:539, during partition_fx_graph_for_cpu_fallback when the compressed_tensors W4A16 quantized linear forward is compiled by the XLA backend.

## Root cause
Two loader bugs were fixed:

1. **transformers 5.x use_fast breaking change (loader)**: `AutoProcessor.from_pretrained("leon-se/gemma-3-27b-it-qat-W4A16-G128")` without `use_fast=False` triggers transformers 5.x to load `Gemma3ImageProcessorFast` despite the checkpoint being saved with slow `Gemma3ImageProcessor`. Fixed by adding `use_fast=False` to `AutoProcessor.from_pretrained()` in `_load_processor()`.

2. **load_shard_spec attribute path wrong (loader)**: In transformers 5.x, `Gemma3ForConditionalGeneration.vision_tower` and `.language_model` are under `.model.vision_tower` and `.model.language_model` respectively (nested inside the inner `Gemma3Model`). The `load_shard_spec` method was accessing them at the wrong level. Fixed by updating the paths to `model.model.vision_tower` and `model.model.language_model`.

After these loader fixes, the remaining blocker is in the tt-xla compiler frontend:

**compressed_tensors W4A16 XLA CPU fallback crash (tt-xla)**: The `leon-se/gemma-3-27b-it-qat-W4A16-G128` model uses `compressed_tensors` W4A16 quantization. When the model's quantized linear layers execute, `compressed_tensors.quantization.lifecycle.forward.quantized_forward` calls underlying linear ops that the XLA compilation pipeline can't lower to TT ops. The `partition_fx_graph_for_cpu_fallback` function in `torch_xla/_dynamo/dynamo_bridge.py` then tries to execute `extract_internal(fused_module)` on the `fused_0` compressed_tensors FusedLinear module. `extract_internal` (line 539) accesses `xla_model.xla_args`, but the `fused_0` module is a plain PyTorch module with no `xla_args` attribute, causing the crash.

## Fix
Loader fixes (committed, pushed):
- `gemma3/multimodal/pytorch/loader.py`: Added `use_fast=False` to `AutoProcessor.from_pretrained()` in `_load_processor()`.
- `gemma3/multimodal/pytorch/loader.py`: Fixed `load_shard_spec` attribute paths from `model.vision_tower` / `model.language_model` to `model.model.vision_tower` / `model.model.language_model`.

Compiler-stack fix (unfixed — Tier B):

The tt-xla `partition_fx_graph_for_cpu_fallback` function in `torch_xla/_dynamo/dynamo_bridge.py` must handle `compressed_tensors` quantized modules (e.g., `FusedLinear`) that do not have `xla_args`. Either:
1. Patch `dynamo_bridge.py:extract_internal` to detect non-XLA fused modules and handle them without `xla_args`; or
2. Teach the tt_torch backend (`python_package/tt_torch/backend/backend.py`) to detect when a sub-graph uses `compressed_tensors` ops and route them to a non-XLA path before `partition_fx_graph_for_cpu_fallback` is invoked.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting

The compressed_tensors W4A16 quantized linear ops are fundamentally incompatible with the current XLA/TT compilation pipeline. Proper support requires either: (a) new lowering patterns for compressed_tensors quantized ops in the XLA pipeline so they can be compiled rather than falling back to CPU, or (b) infrastructure changes to the CPU fallback partition logic to handle non-XLA fused modules. Both are cross-cutting changes that affect all `compressed_tensors`-quantized model variants.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    431.48s (0:07:11) on second run (with loader fixes)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma3/multimodal/pytorch/loader.py`
  - Added `use_fast=False` to `AutoProcessor.from_pretrained()` (commit 2c4eb3a)
  - Fixed `load_shard_spec` attribute paths to use `model.model.*` (commit d96f505)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | af0a828464b394d8db0c974d28ee1c6d725f1823 |
| tt-forge-models | d96f5052c260acd9344242980e9361efb97e0081 |
