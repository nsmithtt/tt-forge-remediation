# Remediation Summary: mradermacher_qwen3_vl_4b_instruct_abliterated_gguf/image_to_text/pytorch-4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_qwen3_vl_4b_instruct_abliterated_gguf/image_to_text/pytorch-4B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — `fast_pos_embed_interpolate` calls `grid_thw.tolist()` on a TT device tensor, which is not supported by the PJRT runtime (INTERNAL: Error code: 13). This is a Tier B `pjrt-device-to-host-transfer` bug requiring new PJRT infrastructure.

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Traceback root (after loader fixes applied):
```
transformers/models/qwen3_vl/modeling_qwen3_vl.py:778: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
transformers/models/qwen3_vl/modeling_qwen3_vl.py:699: in fast_pos_embed_interpolate
    grid_thw_list = grid_thw.tolist()
python_package/tt_torch/torch_overrides.py:34: in __torch_function__
    return func(*args, **(kwargs or {}))
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
**Two loader bugs fixed; one compiler-stack bug unfixed.**

### Loader bug 1: `qwen3vl` GGUF architecture not registered (FIXED)
The mradermacher Qwen3-VL-4B-Instruct-abliterated GGUF file stores
`general.architecture = "qwen3vl"` (no underscore). `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raises `ValueError: GGUF model with architecture qwen3vl is not supported yet.` because `qwen3vl` is absent from `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING["config"]`.

Additionally, when loading weights, `get_gguf_hf_weights_map` uses `hf_model.config.model_type = "qwen3_vl"` (with underscore) which doesn't match the GGUF arch string. The fix: register `qwen3vl` in both tables, then intercept `load_gguf_checkpoint` to remap model_type `"qwen3vl" → "qwen3"` during config loading (so AutoConfig builds a `Qwen3Config` with the right text-backbone fields) and temporarily masquerade `Qwen3VLConfig` as `model_type="qwen3"` during weight loading (so `get_gguf_hf_weights_map` maps `blk.N.*` / `token_embd.*` tensor names to `model.language_model.*`). The full `Qwen3VLConfig` is built by combining the GGUF-derived text config with the vision config from the base `Qwen/Qwen3-VL-4B-Instruct` model.

**Layer: loader (tt-forge-models)**

### Loader bug 2: narrow-signature wrappers overwrite our patch (FIXED)
Several other loaders in the test session install module-level wrappers with
the narrow signature `(gguf_path, return_tensors=False)` — dropping the
`model_to_load` kwarg added in transformers 5.2.0. These module-level
functions store the previous version in `__globals__` (not `__closure__`),
forming a chain that simple BFS via closures cannot traverse. A targeted
BFS walk that also scans `__globals__` for keys containing `"load_gguf"` or
common "orig" names finds the real `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
function (identified by `__module__` and `__qualname__`). The loader then
installs its wrapper at ALL binding sites and re-installs it fresh in
`load_model()` / `_build_full_config()` to survive overwrites by later-imported
loaders.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: `pjrt-device-to-host-transfer` (UNFIXED)
After the loader fixes, `Qwen3VLVisionModel.fast_pos_embed_interpolate` calls
`grid_thw.tolist()` at `modeling_qwen3_vl.py:699`. The test framework has moved
all inputs (including `image_grid_thw`) to the TT device. The TT PJRT runtime
does not implement synchronous device-to-host tensor reads; attempting `.tolist()`
on a TT tensor raises `INTERNAL: Error code: 13`.

**Layer: tt-xla (PJRT runtime)**

## Fix
### Applied (loader, tt-forge-models):

1. **Register `qwen3vl` GGUF architecture**: Add `"qwen3vl"` to `GGUF_SUPPORTED_ARCHITECTURES`
   and `GGUF_TO_TRANSFORMERS_MAPPING["config"]` (copying from the `qwen3` entry).

2. **Patch `load_gguf_checkpoint` with BFS walk**: Define `_find_real_load_gguf_checkpoint()`
   that BFS-walks the monkey-patch chain by checking both `__closure__` cells and
   `__globals__` entries whose names contain `"load_gguf"` or common `"orig"`/`"real"`
   patterns. The wrapper calls the real function directly, bypassing the narrow-signature
   chain. Re-install the patch fresh in `_build_full_config()` and `load_model()` to
   survive overwrites by later-imported loaders.

3. **Build full `Qwen3VLConfig`**: Call `AutoConfig.from_pretrained(gguf_repo, gguf_file=...)` 
   (returns `Qwen3Config` after the architecture remap) for the text backbone, and
   `AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")` for vision config. Combine
   into a `Qwen3VLConfig` and pass as `config=` to `from_pretrained`. Also set
   `ignore_mismatched_sizes=True` for vision encoder weight mismatches.

4. **`requirements.txt`**: Add `gguf>=0.10.0`.

### Proposed (compiler-stack, tt-xla):
Implement synchronous device-to-host tensor value reads in the TT PJRT buffer
layer so that `.tolist()` / `.item()` / `.numpy()` on a TT tensor blocks until
the device computation completes and transfers the result to host. This requires
new infrastructure in the PJRT buffer management path.

## Tier B justification
**Indicator: new-infrastructure**

The PJRT runtime currently has no path for synchronous device-to-host tensor
value reads. Implementing `.tolist()` / `.item()` on TT tensors requires adding
a host-read primitive to the PJRT buffer layer — this is new infrastructure, not
a scoped fix to an existing function.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    272.61s (0:04:32) before failing at fast_pos_embed_interpolate
- Tier A attempts: N/A

## Files changed
- `mradermacher_qwen3_vl_4b_instruct_abliterated_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)
- `mradermacher_qwen3_vl_4b_instruct_abliterated_gguf/image_to_text/pytorch/requirements.txt` (tt-forge-models, new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | db2dc840df6df9439986d64e5f96174fd0883ccf |
| tt-forge-models | 7a2ff1d4684e9abae914a459bbb28e944f39c99e |
