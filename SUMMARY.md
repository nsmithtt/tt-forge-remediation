# Remediation Summary: gigachat-causal_lm-pytorch-3_1-10B-A1_8B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gigachat/causal_lm/pytorch-3.1-10B-A1.8B-single_device-inference]

## Result
FAIL — Tier B compiler bug: MoE expert-weight gather lowered to ttnn::embedding with CB row exceeding L1

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
embedding-rm-cb-weight-row-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Original CI failure (before loader fixes): E   NotImplementedError: "histogram_cpu" not implemented for 'Int'

## Root cause
Two loader bugs were found and fixed; the remaining failure is a Tier B
compiler bug in tt-metal.

**Loader bug 1 — FP8 quantization requires triton:**
`ai-sage/GigaChat3.1-10B-A1.8B` has `quantization_config` (FP8
block-wise, `weight_block_size=[128,128]`) embedded in `config.json`.
transformers 5.2 instantiates `FinegrainedFp8HfQuantizer` which does
`import triton` at module level — triton is not installed in the TT
venv, so loading fails with `ModuleNotFoundError: No module named
'triton'`. Fix: load config, delete `quantization_config`, pass to
`from_pretrained` so weights are cast to BF16.

**Loader bug 2 — histc on integer tensor (original CI failure):**
With FP8 bypassed and model loaded as DeepSeek-V3 MoE, the default MoE
implementation (`grouped_mm_experts_forward`) calls `torch.histc` on an
integer expert-index tensor to build a token-count histogram. XLA does
not support `histogram_cpu` for integer dtypes → `NotImplementedError`.
Fix: set `model.config._experts_implementation = "batched_mm"` after
loading; this selects `batched_mm_experts_forward` which uses
`torch.zeros + scatter_add` instead.

**Tier B compiler bug — embedding CB row exceeds L1:**
After both loader fixes, compilation and execution proceed until the MoE
expert weight gather `gate_up_proj[expert_ids_clamped]` is lowered to
`ttnn::embedding`. The weight tensor has shape `[64, 1966080]` (64
experts × 2×intermediate BF16 elements). Each embedding row is
`1966080 × 2 B = 3.75 MB`, far exceeding the 1.5 MB L1 buffer.
`EmbeddingsRMProgramFactory` sets `out_cb_size = 2 × 3,932,160 =
7.5 MB`; `validate_circular_buffer_region` throws, surfacing as INTERNAL
Error code: 13. The Fused factory has a chunked-processing path, but the
RM factory does not; adding it requires new infrastructure (new program
factory path, reader kernel, writer kernel).

## Fix
**Two loader fixes applied** in
`gigachat/causal_lm/pytorch/loader.py` in `tt_forge_models`:

1. Before `from_pretrained`, load config, delete
   `quantization_config` attribute if present, pass modified config to
   bypass FP8 quantizer.
2. After `from_pretrained`, set
   `model.config._experts_implementation = "batched_mm"` to avoid histc
   on integer tensors.

**Tier B fix (proposed, not implemented):** Add a chunked-processing
path to `EmbeddingsRMProgramFactory` in `tt-metal` (or redirect large
rows to the Fused factory) so that embedding rows larger than L1 can be
processed in tiles rather than failing with CB overflow.

## Tier B justification
Indicator: **new-infrastructure**. `EmbeddingsRMProgramFactory` has no
chunked path. Adding one requires new program factory logic, a new reader
kernel variant, and a new writer kernel variant — cross-cutting changes
within `tt-metal` that go beyond a one-function scoped fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 231.06s (0:03:51)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gigachat/causal_lm/pytorch/loader.py` — bypass FP8
  quantization and set `_experts_implementation = "batched_mm"`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1213e225a8d2939a23e0c6b07366b9aede14540d |
| tt-forge-models | b01382ea3fc0152fd5246a7908e3b681be6471af |
