# Remediation Summary: ltx2_gguf_org-pytorch-19B_dev_Q2_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ltx2_gguf_org/pytorch-19B_dev_Q2_K-single_device-inference]

## Result
XFAIL — 19B model dequantized to BF16 (~37.75 GB) exceeds 32 GB single-device DRAM on p150b

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-parameter-torch-function-dynamo-recursion

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded

## Root cause
Four loader bugs and one Tier A compiler bug:

**Loader bug 1 (config detection → shape mismatch ValueError):** `from_single_file` auto-detects the LTX-2 GGUF as LTX-Video-0.9.7 because the GGUF contains pre-conversion keys (e.g. `patchify_proj.weight`) that match the older model signature. This loads `caption_channels=4096` from Lightricks/LTX-Video-0.9.7-dev instead of the correct `caption_channels=3840` from Lightricks/LTX-2, causing diffusers' shape validator to raise `ValueError: audio_caption_projection.linear_1.weight has an expected quantized shape of: (2048, 3840), but received shape: torch.Size([2048, 7680])`.

**Loader bug 2 (RecursionError — the originally reported failure):** After providing the correct config, `GGUFQuantizationConfig` leaves model weights as `GGUFParameter` instances. `GGUFParameter.__torch_function__` calls `super().__torch_function__()`, which under TorchDynamo tracing recurses infinitely, producing `torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded`.

**Loader bug 3 (incorrect inputs):** `load_inputs` passed `sigma`/`audio_sigma` tensors (unused by LTX-2 forward) and a float-typed `timestep` with shape `(batch_size,)`. The forward expects a `torch.long` timestep with shape `(batch_size, video_seq_len)`.

**Loader bug 4 (apply_split_rotary_emb incompatible with XLA):** `apply_split_rotary_emb` in `diffusers.models.transformers.transformer_ltx2` uses in-place `addcmul_` on tensor views and a non-contiguous `.swapaxes().reshape()`. XLA requires out-of-place operations; `.reshape()` on a non-contiguous tensor also requires `.contiguous()` before the reshape.

**Compiler bug Tier A (prims::view_of alias annotation):** `partition_fx_graph_for_cpu_fallback` runs the FX graph eagerly through `TorchFunctionMode`. Under `_AutoDispatchBelowAutograd`, `prims::view_of.default` is re-dispatched via the Functionalize dispatch key, which rejects ops with alias annotations `Tensor(a)->Tensor(a)`. Fixed by returning `args[0]` directly in `torch_overrides.py` since `prims::view_of` is semantically an identity view.

**Hardware ceiling:** After all fixes, dequantizing 19B parameters to BF16 produces a ~37.75 GB model, which exceeds the 32 GB single-device DRAM of the p150b (Blackhole). The test correctly xfails with an OOM error.

## Fix
**Loader fixes** in `tt_forge_models/ltx2_gguf_org/pytorch/loader.py` (branch `remediation/ltx2_gguf_org-pytorch-19B_dev_Q2_K-single_device-inference` in tt_forge_models):
1. Pass `config="Lightricks/LTX-2", subfolder="transformer"` to `from_single_file` to bypass broken auto-detection.
2. Call `_dequantize_gguf_and_restore_linear(transformer)` then clear `_hf_quantizer=None` and `is_quantized=False` before `.to(dtype)` to prevent the TorchDynamo recursion.
3. Remove `sigma`/`audio_sigma` from `load_inputs`; change `timestep` to `torch.full((B, T), 500, dtype=torch.long)` and `audio_timestep` to `torch.full((B,), 500, dtype=torch.long)`.
4. Patch `diffusers.models.transformers.transformer_ltx2.apply_split_rotary_emb` with an out-of-place XLA-compatible implementation that avoids in-place ops and adds `.contiguous()` before reshape.

**Compiler fix (Tier A)** in `tt_torch/torch_overrides.py` (branch `remediation/ltx2_gguf_org-pytorch-19B_dev_Q2_K-single_device-inference` in tt-xla):
- Return `args[0]` directly when `func is torch.ops.prims.view_of.default` to bypass the alias annotation rejection under the Functionalize dispatch key.

**Test config** in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (same tt-xla remediation branch):
- Mark `ltx2_gguf_org/pytorch-19B_dev_Q2_K-single_device-inference` and `ltx2_gguf_org/pytorch-19B_dev_IQ4_NL-single_device-inference` as `KNOWN_FAILURE_XFAIL` with reason "Out of Memory: 19B model dequantized to BF16 is ~37.75 GB, exceeding 32 GB single-device DRAM capacity".

## Verification
- pytest exit: xfailed (exit code 0, 1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    819.94s (0:13:39) — includes ~7 min wait for device lock from a competing process
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — prims::view_of Tier A fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL for both Q2_K and IQ4_NL variants
- `tt-xla/third_party/tt_forge_models` (submodule pointer → remediation branch)
- `tt_forge_models/ltx2_gguf_org/pytorch/loader.py` — all four loader fixes

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c9602b9981b7471ebfc34781de4810bba0597aff |
| tt-forge-models | 91f1a6d6d98f76f125c3488bd919c2b7b4e5b675 |
