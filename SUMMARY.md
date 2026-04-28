# Remediation Summary: mercuriusdream_huihui_qwen3_5_35b_a3b_abliterated_mlx_2bit_low-causal_lm-pytorch-35B_A3B_Abliterated_MLX_2bit_low-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mercuriusdream_huihui_qwen3_5_35b_a3b_abliterated_mlx_2bit_low/causal_lm/pytorch-35B_A3B_Abliterated_MLX_2bit_low-single_device-inference]

## Result
FAIL — loader cannot read MLX 2-bit affine-quantized model; all weights load as MISSING, model runs with random weights and hangs indefinitely (same root cause as huihui_qwen3_5_35b_a3b_abliterated_4bit_mlx)

## Stack layer
loader

## Tier
B

## Bug fingerprint
mlx-affine-quantized-model-incompatible-format

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

(Original failure message. In our reproduction run the test hung for 10+ minutes without completing or segfaulting; the underlying loader bug was confirmed independently via safetensors inspection.)

## Root cause
The HuggingFace repository `MercuriusDream/Huihui-Qwen3.5-35B-A3B-abliterated-MLX-2bit-low`
stores weights in MLX's native 2-bit affine quantization format, which is fundamentally
incompatible with standard `AutoModelForCausalLM.from_pretrained` loading.

**Three layers of incompatibility** (identical to the 4-bit MLX variant, `huihui_qwen3_5_35b_a3b_abliterated_4bit_mlx`):

1. **Key naming mismatch**: All 1757 checkpoint tensors use a `language_model.*` prefix (e.g.,
   `language_model.model.layers.0.linear_attn.A_log`), while `Qwen3_5MoeForCausalLM` (what
   `AutoModelForCausalLM` resolves to for `Qwen3_5MoeConfig`) expects a plain `model.*`
   namespace. With `ignore_mismatched_sizes=True`, every checkpoint tensor is silently classified
   as UNEXPECTED and every model parameter as MISSING.

2. **Weight format incompatibility**: Quantized weights are stored as `torch.uint32` (16 nibbles
   packed per 32-bit word for 2-bit quantization), e.g. `embed_tokens.weight` has shape
   `[248320, 128]` instead of the expected `[248320, 2048]` bfloat16. Affine dequantization
   parameters are stored in separate `.scales` and `.biases` tensors (shape `[N, M/group_size]`,
   dtype bfloat16, group_size=64) that transformers has no mechanism to use.

3. **Missing dependencies**: Layers 0,1,2 in every group of 4 are `linear_attention`
   (`Qwen3_5MoeGatedDeltaNet`) requiring `flash-linear-attention` and `causal-conv1d` for
   efficient computation. Without them, transformers falls back to a slow sequential
   implementation that substantially lengthens inference time even on random weights.

**Net effect**: The 35B `Qwen3_5MoeForCausalLM` is instantiated with fully random bf16 weights,
then compiled and dispatched to TT hardware. The original segfault is consistent with random
weights triggering pathological values in the Mamba/SSM recurrence (e.g., `exp(-A_log)` on
random floats). In our reproduction run the process ran for 10 minutes without completing.

## Fix
**Proposed fix** (Tier B — not implemented):

The loader `mercuriusdream_huihui_qwen3_5_35b_a3b_abliterated_mlx_2bit_low/causal_lm/pytorch/loader.py`
needs a custom weight-loading path that:

1. Loads the `uint32` tensors from safetensors as raw bytes.
2. Unpacks 2-bit nibbles (16 per uint32 word) into int8 or float32.
3. Applies per-group affine dequantization:
   `w_float[i, j] = w_2bit[i, j] * scales[i, j // group_size] + biases[i, j // group_size]`
   where `group_size = 64`.
4. Strips the `language_model.` prefix from checkpoint keys before assigning to model
   parameters (matching `Qwen3_5MoeForCausalLM`'s `model.*` namespace), or alternatively
   load as `Qwen3_5MoeModel` whose sub-attribute `self.language_model` would accept the
   `language_model.*` prefix directly.
5. Installs `flash-linear-attention` and `causal-conv1d` dependencies in `requirements.txt`.

Even after a correct load, the full bf16 model (35B params × 2 bytes ≈ 70 GB) would exceed
single-device p150b DRAM (24 GB); a hardware-capacity determination would be needed after the
loader is fixed.

## Tier B justification
**new-infrastructure**: MLX's 2-bit affine quantization (uint32 packing + per-group scale/bias)
has no existing support in transformers or in this codebase. Implementing correct 2-bit nibble
unpacking and group-wise dequantization, plus key-namespace remapping for a 1757-tensor
checkpoint, requires new non-trivial infrastructure in the loader. The fix is scoped to one
file but constitutes new infrastructure rather than a one-line patch.

## Verification
- pytest exit: FAIL (process killed by SIGTERM after 10 min; never completed)
- Hardware:    blackhole-p150b
- Duration:    >10:00 wall-clock (terminated by timeout)
- Tier A attempts: N/A

## Files changed
None — Tier B; no code was changed.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
