# Remediation Summary: gemma3-multimodal-pytorch-gaunernst-gemma-3-27b-it-qat-compressed-tensors-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3/multimodal/pytorch-gaunernst/gemma-3-27b-it-qat-compressed-tensors-single_device-inference]

## Result
FAIL — Tier B compiler bug: XLA dynamo bridge fails to compile compressed-tensors `quantized_forward` hooks; `AttributeError: 'fused_0' object has no attribute 'xla_args'` in `partition_fx_graph_for_cpu_fallback`

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
xla-dynamo-compressed-tensors-fused-xla-args

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause

Three cascading loader bugs precede the final compiler bug:

**Loader bug 1 (reported failure)**: `AutoProcessor.from_pretrained()` missing `use_fast=False`. In transformers 5.x, when `use_fast` is not set and the checkpoint was saved with a slow processor (`Gemma3ImageProcessor`), the library warns and loads the fast processor (`Gemma3ImageProcessorFast`) instead. Fix: add `use_fast=False`.

**Loader bug 2**: `compressed-tensors` library not in `requirements.txt`. `gaunernst/gemma-3-27b-it-qat-compressed-tensors` uses compressed-tensors quantization format (`quant_method: "compressed-tensors"`), which requires the `compressed-tensors` Python library. Without it, `Gemma3ForConditionalGeneration.from_pretrained()` raises `ImportError: compressed_tensors is not installed`.

**Loader bug 3**: After installing `compressed-tensors`, the quantization config `ignore` list contains `"re:vision_tower.*"`. The compressed-tensors library uses `re.match()` (anchored at string start) for this pattern. In transformers 5.x, `Gemma3ForConditionalGeneration` wraps submodels inside `self.model` (a `Gemma3Model`), so vision tower modules are now named `model.vision_tower.*` instead of `vision_tower.*`. The unmodified pattern fails to exclude the vision tower from quantization, causing `ValueError: tensor column shape must be divisible by the given group_size 32 but got 4304` (vision tower's `fc2` weight has 4304 columns).

**Compiler bug (Tier B)**: After all loader fixes, the model loads successfully but fails during XLA compilation with `AttributeError: 'fused_0' object has no attribute 'xla_args'`. This occurs in `torch_xla/_dynamo/dynamo_bridge.py::extract_internal` when `partition_fx_graph_for_cpu_fallback` tries to compile fused subgraphs. The compressed-tensors `run_compressed=True` mode installs `quantized_forward` hooks on each quantized linear layer; these hooks access packed 4-bit weight tensors during inference. The `InputCollector.run()` step in `partition_fx_graph_for_cpu_fallback` fails to set `fused_module.xla_args` before `extract_internal(fused_module)` is called, because the quantized_forward ops are unsupported by XLA and cannot be properly partitioned, leaving the fused module's `xla_args` attribute unset.

## Fix
Three loader fixes applied in `gemma3/multimodal/pytorch/loader.py` on branch `remediation/gemma3-multimodal-pytorch-gaunernst-gemma-3-27b-it-qat-compressed-tensors-single_device-inference`:

1. **use_fast=False**: `AutoProcessor.from_pretrained(pretrained_model_name, use_fast=False, **kwargs)` — suppresses the transformers 5.x FutureWarning and ensures the slow processor is loaded consistently.

2. **requirements.txt**: Added `gemma3/multimodal/pytorch/requirements.txt` containing `compressed-tensors` — the library required to load models quantized with the compressed-tensors format.

3. **Vision tower ignore path + load_shard_spec**: For `GEMMA_3_27B_IT_QAT_COMPRESSED_TENSORS`, load the config, and remap `"re:vision_tower.*"` → `"re:model.vision_tower.*"` in `quantization_config["ignore"]` before calling `from_pretrained()`. Also fixed `load_shard_spec` to use `model.model.vision_tower.*` and `model.model.language_model.*` paths (transformers 5.x structural change).

**Proposed compiler fix (not attempted — Tier B)**: The `partition_fx_graph_for_cpu_fallback` function in `torch_xla/_dynamo/dynamo_bridge.py` needs to handle the case where `InputCollector.run()` does not set `xla_args` on a fused module (because the graph contains unsupported ops that prevent execution). Either: (a) guard `extract_internal(fused_module)` to skip modules without `xla_args`, or (b) ensure `InputCollector.run()` succeeds even when the graph contains compressed-tensors quantized ops. The deeper fix is to add proper lowering for compressed-tensors pack-quantized 4-bit dequantization ops in the TT MLIR compiler so they don't need to fall back to CPU.

## Tier B justification
**cross-cutting**: Supporting compressed-tensors `run_compressed=True` inference on TT silicon requires either (a) implementing pack-quantized 4-bit dequantization lowering in the TT MLIR compiler, touching multiple lowering files, or (b) fixing `partition_fx_graph_for_cpu_fallback` in `torch_xla/_dynamo/dynamo_bridge.py` (an upstream Tenstorrent torch-xla build) to handle re-entrant compilation when the model graph contains quantized_forward hooks. Neither is a scoped single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    368.12s (0:06:08) before AttributeError during XLA compilation
- Tier A attempts: N/A

## Files changed
- `gemma3/multimodal/pytorch/loader.py` — use_fast=False, vision_tower ignore path remap, load_shard_spec paths
- `gemma3/multimodal/pytorch/requirements.txt` — new file, adds `compressed-tensors` dependency

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9e00d7fc625aa09ec6f8d59572f2912237367504 |
| tt-forge-models | 7bccab96fa552a00f8301f898c29d437dbc311db |
