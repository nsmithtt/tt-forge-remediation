# Remediation Summary: inferencerlabs_qwen3_5_4b_mlx_9bit-causal_lm-pytorch-4B_MLX_9bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[inferencerlabs_qwen3_5_4b_mlx_9bit/causal_lm/pytorch-4B_MLX_9bit-single_device-inference]

## Result
FAIL — loader bug fixed (MLX affine-8bit dequantization); SSM chunk_gated_delta_rule compilation hangs on TT device (Tier B)

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
Two-layer bug:

**Layer 1 — loader (fixed):** The original loader calls `AutoModelForCausalLM.from_pretrained('inferencerlabs/Qwen3.5-4B-MLX-9bit')` with no special handling for the MLX affine-8bit quantization format. The checkpoint's `config.json` contains a `quantization_config` block with no `quant_method` field, which transformers 5.x cannot interpret. The safetensors checkpoint stores weights as uint32-packed int8 values with per-group bf16 scales/biases under keys prefixed with `language_model.` (e.g. `language_model.model.layers.0.mlp.gate_proj.weight`), while the model created by transformers expects `model.` keys. As a result, `from_pretrained` loads all 426 parameters as UNEXPECTED/MISSING: the model receives only random float32 weights. CPU inference on this uninitialized 4B fp32 model with 24 SSM (GatedDeltaNet) layers using the pure-PyTorch fallback for `chunk_gated_delta_rule` takes 10+ minutes, causing the test timeout.

**Layer 2 — tt-mlir (unfixed, Tier B):** After fixing the loader, the model loads correctly in bf16 and CPU inference completes in seconds. However, the `Qwen3_5GatedDeltaNet.forward` method calls `torch_chunk_gated_delta_rule` (the pure-PyTorch fallback for the `fla-org/flash-linear-attention` library). This fallback contains two Python `for` loops — one over `chunk_size=64` iterations performing triangular inplace updates (`attn[..., i, :i] = ...`), and one over `total_sequence_length // chunk_size` chunks. When XLA traces the model, it unrolls these loops, producing a large StableHLO graph with hundreds of scatter/update operations. The TT MLIR compiler subsequently hangs compiling this graph — the compilation never completes within the test timeout (observed 56+ minutes on silicon with the process consuming ~70 GB RAM and 183% CPU before being killed).

## Fix
**Loader fix (applied):** The existing remediation branch `remediation/inferencerlabs_qwen3_5_4b_mlx_9bit-causal_lm-pytorch-4B_MLX_9bit-single_device-inference` at commit `d9552c7eb4` in tt_forge_models contains the correct fix:
- `AutoConfig.from_pretrained` fetches the config; `outer_config.text_config` is used to create the base `Qwen3_5ForCausalLM` on meta device via `from_config`, bypassing the unsupported `quantization_config`.
- `hf_hub_download` + `safetensors.load_file` loads the raw checkpoint.
- `_dequantize_mlx_affine` dequantizes each uint32-packed weight (`w.view(uint8).view(int8) * scales + biases`), skips `.scales`/`.biases` entries, strips the `language_model.` prefix, and permutes 3D Conv1d weights from MLX channel-last `[out, kernel, in]` to PyTorch `[out, in, kernel]`.
- `model.load_state_dict(..., strict=False, assign=True)` + `tie_weights()` + `Qwen3_5TextRotaryEmbedding` re-init complete the load. All 426 parameters match.

**Compiler fix (proposed, Tier B):** The proper fix lives in tt-mlir. The StableHLO graph emitted by unrolled SSM scan loops must either be (a) recognized and lowered as a scan primitive in TTIR/TTNN, or (b) handled via a size-adaptive loop-fusion pass that avoids materializing each loop iteration as separate ops. This requires new infrastructure in the MLIR lowering pipeline.

## Tier B justification
Which indicator: new-infrastructure

The `torch_chunk_gated_delta_rule` Python loops unroll into hundreds of chained scatter/update ops in StableHLO. Handling scan-like patterns efficiently requires either a new scan primitive in TTIR (new op + lowering) or a loop-recognition and fusion pass in the MLIR pipeline. Neither is a scoped one-function tweak; both require significant new infrastructure in tt-mlir.

## Verification
- pytest exit: TIMEOUT
- Hardware:    blackhole-p150b
- Duration:    56+ min (killed)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/inferencerlabs_qwen3_5_4b_mlx_9bit/causal_lm/pytorch/loader.py` — rewritten to dequantize MLX affine-8bit weights (commit `d9552c7eb4` on the tt_forge_models remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d9552c7eb42f3ec33809b3370a86a419e19ee97a |
