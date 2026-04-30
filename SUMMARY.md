# Remediation Summary: deepseek_r1_distill_qwen_7b_4bit-causal_lm-pytorch-DeepSeek_R1_Distill_Qwen_7B_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_7b_4bit/causal_lm/pytorch-DeepSeek_R1_Distill_Qwen_7B_4bit-single_device-inference]

## Result
FAIL — loader fix applied; PCC=0.9627 < 0.99 after dequantization; WH BF16 matmul precision floor

## Stack layer
loader, tt-mlir

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
```
ValueError: The model's quantization config from the arguments has no `quant_method` attribute.
Make sure that the model has been correctly quantized
```

## Root cause
Two separate issues:

**Loader bug (fixed):** `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` stores weights in MLX
native 4-bit format. The `config.json` contains `quantization_config: {"group_size": 64, "bits": 4}` —
an MLX-format dict with no `quant_method` field. Transformers 5.x `get_hf_quantizer` calls
`AutoHfQuantizer.supports_quant_method(config.quantization_config)`, which raises `ValueError`
because `hasattr(quantization_config, "quant_method")` returns False for a plain dict. The weights
themselves are stored as packed `uint32` (8 int4 values per element, LSB-first) with companion
`.scales` and `.biases` tensors — they cannot be loaded as regular float tensors without
dequantization.

**Compiler precision (unfixed, Tier B):** After dequantizing the weights to bfloat16, the model
compiles and runs on TT silicon, but PCC=0.9627 vs CPU FP32 reference, below the required 0.99.
CPU BF16 vs FP32 PCC is 0.9836, also below 0.99, establishing a BF16 precision floor inherent to
this 7B model. The additional TT-specific degradation (0.9836 → 0.9627, Δ=0.021) is consistent
with the known WH BF16 matmul precision issue observed in Gemma 7B (PCC=0.915), Qwen3 4B
(PCC=0.864), GPT-J 6B (PCC=0.75), and BlackSheep 12B (PCC=0.949).

## Fix
**Loader fix** in `tt_forge_models/deepseek_r1_distill_qwen_7b_4bit/causal_lm/pytorch/loader.py`:
- Load config with `AutoConfig.from_pretrained`, then set `config.quantization_config = None` to
  strip the MLX quantization dict before passing to the transformers quantizer path.
- Initialize model with `AutoModelForCausalLM.from_config(config)` (no pretrained weights).
- Load `model.safetensors` directly via `safetensors.safe_open`. For each layer whose `.weight` key
  has a `.scales` sibling, dequantize via `_mlx_dequantize`: unpack 8 uint4 values per uint32 element
  (LSB-first), then apply per-group scales and biases (`out = int4 * scale + bias`). Plain float tensors
  (layer norms, attention biases) are passed through with `tensor.to(target_dtype)`.
- Load the resulting state dict with `model.load_state_dict(state_dict, strict=False)`.
- MLX keys map directly to transformers Qwen2 state-dict keys — no key remapping required.

**Proposed compiler fix (Tier B):** The WH BF16 matmul precision floor requires either F32
accumulation paths in tt-mlir (cross-cutting, many lowering passes) or a hardware-level fix in
tt-metal. Filed as the same `ttmlir-bf16-matmul-precision-floor` bug class as other 7B+ models.

## Tier B justification
`cross-cutting`: Raising BF16 matmul accumulation fidelity to match CPU FP32 precision requires
changes to every matmul lowering in tt-mlir and tt-metal, not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~6 minutes (5:20 compile + 0:04 inference)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/deepseek_r1_distill_qwen_7b_4bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b93d9e96bfb3f12f86993592c65b38165d920821 |
| tt-forge-models | 1d84bd3a1276f78faf4112eadbf17dd62b58a6b7 |
