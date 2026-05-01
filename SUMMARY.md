# Remediation Summary: jackrong_qwen3_5_9b_claude_reasoning_abliterated_fp16_mlx-causal_lm-pytorch-9B_Claude_Reasoning_Abliterated_fp16_MLX-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jackrong_qwen3_5_9b_claude_reasoning_abliterated_fp16_mlx/causal_lm/pytorch-9B_Claude_Reasoning_Abliterated_fp16_MLX-single_device-inference]

## Result
FAIL — SSM scan loop (GatedDeltaNet) compilation hangs TT MLIR compiler; Tier B new-infrastructure

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ssm-scan-loop-compilation-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
Two bugs were found in series:

**Bug 1 (loader — fixed):** The checkpoint `AITRADER/Jackrong-Qwen3.5-9B-Claude-Reasoning-abliterated-fp16-MLX`
was saved as `Qwen3_5ForConditionalGeneration` (VLM) and stores text weights under a
`language_model.*` prefix (`language_model.model.layers.*`, `language_model.lm_head.*`).
`AutoModelForCausalLM.from_pretrained` maps `Qwen3_5Config` → `Qwen3_5ForCausalLM` (text-only,
expects flat `model.*` / `lm_head.*` keys). The auto-factory does extract `text_config` before
calling `from_pretrained`, so model instantiation succeeds, but none of the checkpoint keys match
the model's expected layout — all weights are MISSING and the model initialises with random values.

**Bug 2 (tt-mlir — unfixed, Tier B):** The model is a Qwen3.5 SSM/Mamba-attention hybrid with 24
`linear_attention` (GatedDeltaNet) layers and 8 full-attention layers. `flash-linear-attention` and
`causal-conv1d` are not installed, so `Qwen3_5GatedDeltaNet.forward` falls back to
`torch_chunk_gated_delta_rule` (pure-Python). XLA traces the two nested Python `for` loops (one
over `chunk_size=64` iterations with triangular in-place slice updates, one over sequence chunks),
unrolling them into hundreds of chained `scatter/dynamic-update-slice` ops in StableHLO. The TT
MLIR compiler hangs on this enormous graph (observed 56+ min consuming 70 GB RAM); the test is
killed at the 30-minute timeout.

## Fix
**Bug 1 loader fix** committed to
`remediation/jackrong_qwen3_5_9b_claude_reasoning_abliterated_fp16_mlx-causal_lm-pytorch-9B_Claude_Reasoning_Abliterated_fp16_MLX-single_device-inference`
in `tt-forge-models` (commit `67e8720919763e7396ed176aac7db79c1e9cb776`):

- `jackrong_qwen3_5_9b_claude_reasoning_abliterated_fp16_mlx/causal_lm/pytorch/loader.py`:
  replace `AutoModelForCausalLM.from_pretrained` with manual safetensors loading.
  Use `snapshot_download` to fetch the shards, strip the `language_model.` prefix from
  all matching keys, and call `Qwen3_5ForCausalLM(text_config).load_state_dict(state_dict, strict=False)`.

**Bug 2 proposed fix (Tier B):** The TT MLIR compiler needs a scan/loop-recognition pass that
fuses the triangular GatedDeltaNet update pattern into a single TTIR scan primitive, rather than
emitting one scatter op per loop iteration. This is a new compilation pass in `tt-mlir`.

## Tier B justification
new-infrastructure: fixing the SSM scan loop requires a new scan-primitive or loop-fusion pass
in TT MLIR to handle the pattern emitted by `torch_chunk_gated_delta_rule`. This is not a bounded
change to one or two files — it requires new TTIR ops and corresponding lowering rules.

## Verification
- pytest exit: TIMEOUT
- Hardware:    blackhole-p150b
- Duration:    >1800s (killed)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: jackrong_qwen3_5_9b_claude_reasoning_abliterated_fp16_mlx/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 67e8720919763e7396ed176aac7db79c1e9cb776 |
