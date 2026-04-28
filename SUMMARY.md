# Remediation Summary: fxmarty_qwen1_5_moe_a2_7b_chat_w_fp4_a_fp6_e2m3-causal_lm-pytorch-A2.7B-Chat-w-fp4-a-fp6-e2m3-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[fxmarty_qwen1_5_moe_a2_7b_chat_w_fp4_a_fp6_e2m3/causal_lm/pytorch-A2.7B-Chat-w-fp4-a-fp6-e2m3-single_device-inference]

## Result
FAIL — AMD Quark version incompatibility: Quark 0.10 (used to quantize model) is incompatible with PyTorch 2.7.0; Quark 0.11 installs but transformers 5.2.0's QuarkHfQuantizer fails to load weight scales (loader bug), and Quark 0.11's scale shape for E8M0 format is wrong

## Stack layer
loader

## Tier
B

## Bug fingerprint
quark-scale-load-broken-transformers5

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

(Reproduced locally as ImportError / RuntimeError through the following loading chain)

## Root cause
The model `fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3` uses AMD Quark 0.10 FP4/FP6
quantization (`quant_method: quark`, `weight_format: real_quantized`, `scale_format: e8m0`).
Three compounding failures in the loader layer prevent this model from loading on the current
stack (PyTorch 2.7.0 / transformers 5.2.0):

1. **Quark 0.10 / PyTorch 2.7.0 incompatibility.** `amd-quark==0.10` (the version matching the
   model's training environment) fails to import because PyTorch 2.7.0 removed
   `torch.onnx._internal.jit_utils`, which `quark.torch.kernel` imports at module level.

2. **transformers 5.2.0 QuarkHfQuantizer broken scale loading.** Installing
   `amd-quark==0.11` resolves the import. However, `QuarkHfQuantizer.get_weight_conversions()`
   in `transformers/quantizers/quantizer_quark.py` registers `WeightConverter` objects with
   `source_patterns=["weight_scale"]` and `target_patterns=["weight_scale"]` (identical).
   Because the renamed key equals the original key, and because `weight_scale` is not present
   in the Quark 0.11 model's meta state dict (the model has `weight_quantizer.scale`, not
   `weight_scale`), the loading framework marks all `weight_scale` checkpoint tensors as
   UNEXPECTED and never calls `QuarkDeserialize.convert()`. The result is that every
   `weight_quantizer.scale` parameter in the loaded model is uninitialized (all zeros).

3. **Quark 0.11 E8M0 scale buffer shape mismatch.** Even if the scales were assigned,
   `_map_to_quark()` (Quark 0.11) creates `weight_quantizer.scale` with shape `[out, n_groups]`
   float32 (e.g. `[2048, 64]` for q_proj with 64 groups). The `unpack_params()` path for
   `scale_format="e8m0"` reinterprets this as `scale.view(torch.uint8)` → `[2048, 256]` uint8
   (4× the E8M0 values actually needed), producing a tensor size mismatch in
   `dequantize_fp4_fp6_per_group`: `inputs` has 64 groups but decoded scale has 256.
   The correct packed scale shape would be `[2048, 16]` float32 (so that 16×4 = 64 uint8 E8M0
   bytes are decoded to 64 group scales). Quark 0.11's `infer_packed_shape` does not account
   for the 4-byte-per-float32 packing factor when computing the scale buffer size for E8M0.

The original segfault on the CI host was almost certainly the forward-pass consequence of the
uninitialized scale (or the Qwen2-MoE XLA for-loop routing issue — not reachable in our
environment because loading itself fails).

## Fix
The fix spans two packages and cannot be done as a single scoped loader change:

1. **transformers `quantizers/quantizer_quark.py`**: Change `get_weight_conversions()` to use
   `target_patterns=["weight_quantizer.scale"]` instead of `target_patterns="weight_scale"` so
   that `rename_source_key()` correctly renames checkpoint `weight_scale` keys to the model's
   `weight_quantizer.scale` keys and they pass the `renamed_key in meta_model_state_dict` check.
   `QuarkDeserialize.convert()` must also be updated to account for the new rename path.

2. **amd-quark `export/nn/modules/qparamslinear.py`** (or the packer's `infer_packed_shape`):
   Fix the scale buffer shape for E8M0 format to be `[out, n_groups // 4]` float32 (packed) so
   that `scale.view(torch.uint8)` yields exactly `n_groups` E8M0 bytes per row.

3. The model loader `requirements.txt` should add `amd-quark` (without version pin once the
   upstream bugs are fixed).

## Tier B justification
cross-cutting — The fix requires coordinated changes in two third-party packages
(`transformers` and `amd-quark`), touching at minimum 3 files across 2 repos. The
QuarkHfQuantizer loading bug is in transformers; the E8M0 scale shape bug is in Quark. Neither
can be fixed independently: fixing transformers without fixing Quark leaves a shape mismatch;
fixing Quark without fixing transformers leaves scales unloaded.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
None — no code changes made (Tier B, fix requires upstream changes in transformers and amd-quark)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
