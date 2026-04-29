# Remediation Summary: deepseek_v3_2_awq-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_2_awq/pytorch-single_device-inference]

## Result
FAIL — tt-mlir lowers 3D expert gather (stablehlo.gather on 256×4096×1024 weight) to ttnn.embedding with 4 MB weight-page CB, overflowing 1.5 MB L1

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-3d-gather-large-page-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Underlying fatal (from device log):
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=9)] grow to 4305920 B which is beyond max L1 size of 1572864 B

## Root cause
Three issues were found in sequence:

**Issue 1 (loader — fixed):** `QuantTrio/DeepSeek-V3.2-AWQ` uses
`model_type: deepseek_v32` in its `config.json`. This type is not registered
in transformers 5.2.0 (which only has `deepseek_v2`, `deepseek_v3`,
`deepseek_vl`, `deepseek_vl_hybrid`). The model repo has no `auto_map`
entries and no custom Python files in the HuggingFace cache, so
`trust_remote_code=True` could not resolve it. `AutoConfig.from_pretrained`
raised `ValueError: model type 'deepseek_v32' not recognized`.

Fix: added `deepseek/deepseek_v3_2_awq/pytorch/src/model_utils.py` which
registers `DeepseekV32Config` (subclass of `DeepseekV3Config`) and
`DeepseekV32ForCausalLM` (subclass of `DeepseekV3ForCausalLM`) in
`AutoConfig` and `AutoModelForCausalLM`. Imported at loader module level
so registration happens before `AutoConfig.from_pretrained`.

**Issue 2 (loader — fixed):** The original failure message (`histogram_cpu
not implemented for 'Int'`) is produced by `grouped_mm_experts_forward` in
`transformers/integrations/moe.py`. Under TT's XLA device
(`device.type == "xla"`), the branch that checks `device.type == "cpu"`
selects `torch.int` for the `torch.histc` input, but CPU histc rejects
integer input. Fix: set `config._experts_implementation = "batched_mm"` in
the loader so the static-shape `batched_mm_experts_forward` path is used
instead.

**Issue 3 (tt-mlir — unfixed, Tier B):** The `batched_mm` path performs
expert lookup via advanced indexing: `gate_up_proj[expert_ids_clamped]` where
`gate_up_proj` has shape `[n_routed_experts, 2*moe_intermediate_size, hidden]`
= `[256, 4096, 1024]` (with test-time reduced dims). tt-mlir lowers this
`stablehlo.gather` by:
1. Reshaping the 3D weight to `[256, 4194304]` (flatten last two dims).
2. Calling `ttnn.embedding` with indices `[10, 1]` against the `[256, 4194304]`
   weight table.

The embedding CB allocates one full row of the weight table as its page:
`4194304 elements × 2 bytes = 8 MB`, which is 5× the 1.5 MB L1 maximum.
This causes the fatal `TT_THROW` at `program.cpp:1136` and surfaces as
`INTERNAL: Error code: 13`. Same bug as `deepseek_v3_2_4bit_mlx` (CB size
4305920 B in both cases, same architecture).

## Fix
**Loader fix (committed)** in
`tt-xla/third_party/tt_forge_models` on branch
`remediation/deepseek-deepseek_v3_2_awq-pytorch-single_device-inference`:

- New file: `deepseek/deepseek_v3_2_awq/pytorch/src/__init__.py`
- New file: `deepseek/deepseek_v3_2_awq/pytorch/src/model_utils.py` —
  registers `deepseek_v32` as alias for `DeepseekV3Config` /
  `DeepseekV3ForCausalLM` in the AutoModel registry.
- Modified: `deepseek/deepseek_v3_2_awq/pytorch/loader.py` — imports
  `model_utils` at module level, sets `config._experts_implementation =
  "batched_mm"` before `from_config`, removes now-redundant
  `trust_remote_code=True`.

**Proposed compiler fix (not attempted, Tier B):** In tt-mlir's
`StableHLOToTTIR` or `TTIRToTTNN` lowering for `stablehlo.gather` on 3D
operands: instead of flattening to a 2D embedding lookup (which inflates
the CB row size to `hidden*intermediate` elements), lower to a sequence of
`ttnn.matmul` or use a proper 3D gather implementation that pages the weight
tensor in smaller tiles. Alternatively, detect when the CB page size would
exceed L1 and fall back to a loop over the expert dimension. This would
touch ≥3 files across tt-mlir and possibly tt-metal (matching kernel changes).

## Tier B justification
- **cross-cutting**: Fixing the 3D gather→embedding lowering requires
  changes to both the MLIR lowering pass (StableHLOToTTIR / TTIRToTTNN)
  and the tt-metal embedding kernel to handle the non-standard per-row
  stride, touching ≥3 files across ≥2 repos.
- **more-than-3-files**: The lowering pass, the embedding program factory,
  and potentially the weight preparation step would all need coordinated
  changes.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    194.15s (0:03:14) to reach INTERNAL error
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_2_awq/pytorch/loader.py` (modified)
- `deepseek/deepseek_v3_2_awq/pytorch/src/__init__.py` (created)
- `deepseek/deepseek_v3_2_awq/pytorch/src/model_utils.py` (created)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6709cdcb371dae08ef176aaf7bc4635cf4575dd2 |
| tt-forge-models | 2dce874f469f71acc5d4073716b7ade1ccc0189d |
