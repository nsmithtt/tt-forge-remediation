# Remediation Summary: jsl_vl_7b_medagentbench_v2_i1_gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jsl_vl_7b_medagentbench_v2_i1_gguf/image_to_text/pytorch-7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Tier B pjrt-device-to-host-transfer bug: rot_pos_emb calls grid_thw.tolist() on a TT tensor (INTERNAL: Error code: 13)

## Stack layer
loader, tt-xla

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
The original failure message was:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.
```
(appeared in stdout during model load, followed by):
```
ValueError: GGUF model with architecture qwen2vl is not supported yet.
```

After loader fixes, the terminal failure is:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
at `grid_thw.tolist()` inside `rot_pos_emb` in `modeling_qwen2_5_vl.py:375`.

## Root cause
Three loader bugs were fixed before reaching the Tier B compiler bug:

1. **qwen2vl GGUF architecture not registered**: transformers 5.x GGUF_SUPPORTED_ARCHITECTURES does not include `qwen2vl` (the architecture field in Qwen2-VL/Qwen2.5-VL GGUF files). Fixed by registering it with the qwen2 config-key mapping.

2. **Broken patcher chain drops model_to_load**: ~26 GGUF loader modules patch `load_gguf_checkpoint` at import time but drop the `model_to_load` kwarg. When `from_pretrained` calls `load_gguf_checkpoint(return_tensors=True, model_to_load=dummy_model)`, the chain drops `model_to_load`, causing `get_gguf_hf_weights_map(None, processor)` → AttributeError. Fixed by a context manager that bypasses the chain and calls the real transformers function directly.

3. **VL-specific rope_parameters (mrope_section) missing from GGUF config**: The GGUF config mapping only carries text-transformer keys; `mrope_section` and other VL RoPE fields are absent. Fixed by loading `AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")` and passing it as `config=` to `from_pretrained` so the full VL config is present.

4. **use_fast=False for processor**: transformers 5.x now defaults Qwen2VLImageProcessor to fast mode; `AutoProcessor.from_pretrained` emits a warning/error. Fixed by adding `use_fast=False`.

After all four loader fixes the model loads and is placed on the TT device. During the forward pass, `rot_pos_emb` in `Qwen2_5_VLModel` calls `grid_thw.tolist()` on a TT tensor at inference time. The TT PJRT backend does not support device-to-host transfer via `.tolist()` and raises `INTERNAL: Error code: 13`. This is the `pjrt-device-to-host-transfer` Tier B infrastructure bug.

## Fix
Loader fixes committed to `remediation/jsl_vl_7b_medagentbench_v2_i1_gguf` branch in tt-forge-models:
- `jsl_vl_7b_medagentbench_v2_i1_gguf/image_to_text/pytorch/loader.py`

Proposed Tier B fix (not implemented): The `pjrt-device-to-host-transfer` bug requires the TT PJRT runtime to support eager device-to-host tensor transfer for scalar/small tensors. The call site is `modeling_qwen2_5_vl.py:375`: `for t, h, w in grid_thw.tolist()`. This would need either (a) a CPU fallback for `.tolist()` in the TT PJRT host-callback path, or (b) the compilation pipeline to detect `.tolist()` and emit a synchronous transfer before the call. This is new infrastructure in `tt-xla` / PJRT runtime.

## Tier B justification
Indicator: **new-infrastructure** — device-to-host transfer via `.tolist()` on TT tensors is not implemented in the PJRT runtime. The fix requires adding a new synchronous transfer path, not a scoped pattern change in a single file.

## Verification
- pytest exit: FAIL
- Hardware: wormhole (blackhole-p150b)
- Duration: ~525s (model loaded, inference reached device, then Error 13)
- Tier A attempts: N/A

## Files changed
- `jsl_vl_7b_medagentbench_v2_i1_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 2854d7eb389341db49e763d1eb020dd10d1d775a |
