# Remediation Summary: cyankiwi_qwen3_vl_8b_instruct_awq_8bit-pytorch-8b_instruct_awq_8bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cyankiwi_qwen3_vl_8b_instruct_awq_8bit/pytorch-8b_instruct_awq_8bit-single_device-inference]

## Result
FAIL — grid_thw.tolist() on XLA lazy tensor synchronisation fails with INTERNAL: Error code: 13 inside Qwen3VL fast_pos_embed_interpolate; loader ImportError (missing compressed-tensors) was fixed but a Tier B tt-xla bug blocks the test

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
qwen3vl-grid-thw-tolist-xla-lazy-sync

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (ImportError, now fixed by loader change):
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization.
Please install it with `pip install compressed-tensors`.
```

Remaining failure after loader fix (162.36s run on n150):
```
venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:699: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two bugs were found:

**Bug 1 (loader, fixed):** The model `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-8bit` uses
`compressed_tensors` pack-quantized int8 format. Without the `compressed-tensors` package
the quantization config fails to instantiate, raising ImportError before the model reaches
silicon. TT hardware has no int8 matmul path so the model must be dequantized to bfloat16
at load time via `run_compressed=False` in the quantization config. The `compressed-tensors
0.15.x` library also attaches instance-level `forward` overrides to quantized Linear modules
that access `weight.data` unconditionally, conflicting with TT-XLA's `__torch_function__`
during `torch.compile`. The loader also had `dtype=` (invalid kwarg) instead of
`torch_dtype=` in the fallback branch.

**Bug 2 (tt-xla, Tier B, unfixed):** After the loader fix, the Qwen3VL visual encoder's
`fast_pos_embed_interpolate` calls `grid_thw.tolist()` on `image_grid_thw`, which the test
runner has placed on the TT device. Under `torch.compile` with the XLA backend,
`image_grid_thw` is a lazy XLA tensor (part of a pending computation graph). Calling
`.tolist()` on a lazy XLA tensor forces synchronisation of that computation graph via the
PJRT backend. This synchronisation fails with `INTERNAL: Error code: 13`. The
`TorchFunctionOverride.__torch_function__` in `torch_overrides.py` intercepts the call and
propagates the error unchanged. A second identical issue exists in `Qwen3VLModel.get_rope_index`
which also calls `.tolist()` for data-dependent RoPE index computation. The alinvlm v1.3
remediation (same 7B Qwen3VL architecture) documented that even with the visual encoder and
RoPE patched, the 8B language model backbone overflows TT L1 (~8 MB needed vs 1.5 MB max).

## Fix
**Loader fix (committed to tt-forge-models, branch remediation/cyankiwi_qwen3_vl_8b_instruct_awq_8bit-pytorch-8b_instruct_awq_8bit-single_device-inference):**
- `cyankiwi_qwen3_vl_8b_instruct_awq_8bit/pytorch/requirements.txt` (created): `compressed-tensors`
- `cyankiwi_qwen3_vl_8b_instruct_awq_8bit/pytorch/loader.py` (updated):
  - Import `AutoConfig`; load config before `from_pretrained` and set
    `quantization_config.run_compressed = False` so int8 pack-quantized weights are
    dequantized to bfloat16 at load time
  - Remove all instance-level `forward` overrides that `compressed-tensors 0.15.x`
    attaches to quantized Linears after decompression (restores class-level `forward`)
  - Set `model.config.use_cache = False` to suppress KV-cache output tensors
  - Fix `dtype=` → `torch_dtype=` in the else-branch of dtype_override handling

**Proposed fix for Tier B bug (not attempted):**
`python_package/tt_torch/torch_overrides.py`: extend `TorchFunctionOverride.__torch_function__`
to detect `.tolist()` / `.item()` calls on XLA/TT device tensors and use a supported
PJRT host-read path to materialise the concrete data before calling the Python-level
operation. This requires either an eager PJRT read API for lazy XLA tensors, or a
mechanism to insert graph breaks so data-dependent Python control flow exits the compiled
graph and resumes eagerly. Patching the model loader's `get_image_features` or
`get_rope_index` to move `image_grid_thw` to CPU before the `.tolist()` call would paper
over the compiler bug and is not an acceptable substitute.

## Tier B justification
`internal-error-unknown-mechanism` — The failure is `INTERNAL: Error code: 13` from the TT
PJRT backend when a lazy XLA tensor is synchronised inside `torch.compile`. The mechanism
inside the XLA execution engine is unknown without PJRT/TT-metal investigation. Fixing the
first `.tolist()` call would expose an identical second call in `get_rope_index`, and the
alinvlm precedent shows the 8B language model backbone would still overflow TT L1 even with
both data-dependent ops resolved. The full fix requires coordinated changes across
`torch_overrides.py` (graph-break/eager-read plumbing) and possibly the PJRT implementation.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    162.36s (0:02:42)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `cyankiwi_qwen3_vl_8b_instruct_awq_8bit/pytorch/loader.py`
- `tt-forge-models`: `cyankiwi_qwen3_vl_8b_instruct_awq_8bit/pytorch/requirements.txt`
- `tt-xla`: `third_party/tt_forge_models` submodule pointer updated to remediation commit

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ec4d400c4 (report/cyankiwi_qwen3_vl_8b_instruct_awq_8bit branch) |
| tt-forge-models | a8015e6d96 (remediation/cyankiwi_qwen3_vl_8b_instruct_awq_8bit branch) |
