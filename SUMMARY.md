# Remediation Summary: huihui_qwen3_5_35b_a3b_abliterated_4bit_mlx-causal_lm-pytorch-35B_A3B_Abliterated_4bit_MLX-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_35b_a3b_abliterated_4bit_mlx/causal_lm/pytorch-35B_A3B_Abliterated_4bit_MLX-single_device-inference]

## Result
FAIL — loader cannot read MLX affine-quantized model; all weights load as MISSING, model runs with random weights and segfaults/hangs on silicon

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

## Root cause
The HuggingFace repository `AITRADER/Huihui-Qwen3.5-35B-A3B-abliterated-4bit-MLX` stores model
weights in MLX's native 4-bit affine quantization format, which is fundamentally incompatible
with standard `AutoModelForCausalLM.from_pretrained` loading.

**Three layers of incompatibility** (confirmed by inspecting the safetensors index):

1. **Key naming mismatch**: All 2090 checkpoint tensors use a `language_model.*` or
   `vision_tower.*` prefix (e.g.,
   `language_model.model.layers.0.linear_attn.A_log`), while `Qwen3_5MoeForCausalLM` expects
   a plain `model.*` namespace. With `ignore_mismatched_sizes=True`, every checkpoint tensor is
   silently classified as UNEXPECTED and every model parameter as MISSING.

2. **Weight format incompatibility**: Quantized weights are stored as `torch.uint32` (8 nibbles
   packed per word), shape `[N, M/8]` instead of the expected `[N, M]` float16/bfloat16. The
   affine dequantization parameters are stored in separate `.scales` and `.biases` tensors (shape
   `[N, M/group_size]`, dtype bfloat16) which transformers has no mechanism to use.

3. **Missing dependencies**: The model uses hybrid Mamba/linear-attention layers
   (`linear_attn.*`) that require `flash-linear-attention` and `causal-conv1d` for efficient
   computation. Without them, transformers falls back to a slow sequential PyTorch implementation.

**Net effect**: The 35B `Qwen3_5MoeForCausalLM` is instantiated with fully random bf16 weights,
then compiled and dispatched to TT hardware. The random-weight model can crash inside the Mamba
SSM recurrence or during a device→host transfer, producing the observed `Fatal Python error:
Segmentation fault`.

## Fix
**Proposed fix** (Tier B — not implemented):

The loader `huihui_qwen3_5_35b_a3b_abliterated_4bit_mlx/causal_lm/pytorch/loader.py` needs a
custom weight-loading path that:

1. Loads the uint32 tensors from safetensors as raw bytes.
2. Unpacks 4-bit nibbles (8 per uint32 word) into int8 or float32.
3. Applies per-group affine dequantization:
   `w_float[i, j] = w_4bit[i, j] * scales[i, j // group_size] + biases[i, j // group_size]`
   where `group_size = 64`.
4. Strips the `language_model.` prefix from checkpoint keys before assigning to model
   parameters.
5. Drops `vision_tower.*` keys since `Qwen3_5MoeForCausalLM` has no vision head.

Additionally, `requirements.txt` would need appropriate fallback CPU implementations of
`flash-linear-attention` ops so the model can be loaded and executed.

Even after a correct load, the full bf16 model (~70 GB) may exceed single-device p150b DRAM;
a hardware-capacity determination would be needed after the loader is fixed.

## Tier B justification
**new-infrastructure**: MLX's 4-bit affine quantization (uint32 packing + per-group scale/bias)
has no existing support in transformers or in this codebase. Implementing correct nibble
unpacking and group-wise dequantization, plus key-namespace remapping for a 2090-tensor
checkpoint, requires new non-trivial code in the loader. The fix is scoped to one file but
constitutes new infrastructure rather than a one-line patch.

## Verification
- pytest exit: not-run (Tier B; loader issue confirmed by safetensors index inspection)
- Hardware:    not-run
- Duration:    N/A
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
