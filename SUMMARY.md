# Remediation Summary: llava_onevision_1_5-pytorch-8B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision_1_5/pytorch-8B_Instruct-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: rot_pos_emb iterates over TT-device integer tensor (image_grid_thw) inside the compilation context; any D2H transfer fails with INTERNAL Error code: 13

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Triggered inside `rot_pos_emb` of the LLaVA-OneVision-1.5 visual encoder. The model's
`rot_pos_emb` method iterates over `grid_thw` (the image grid dimensions T,H,W) with
Python control flow to build position indices. When `grid_thw` is on the TT device,
any attempt to access its values — directly via iteration yielding TT 0-dim tensors
as `torch.arange(h)` arguments, or via explicit `.cpu()` / `.tolist()` calls — triggers
a device-to-host transfer that fails inside the compiled region.

## Root cause
The LLaVA-OneVision-1.5 visual encoder's `rot_pos_emb` and `get_rope_index` methods
contain data-dependent Python control flow over integer metadata tensors (`image_grid_thw`,
`input_ids`). The test framework moves all model inputs, including integer metadata, to the
TT device before calling the model under torch.compile. Inside the compiled context, every
tensor operation (including `.cpu()` and iteration that yields 0-dim TT tensors to
`torch.arange`) is traced by Dynamo into the StableHLO graph. The resulting graph includes
device-to-host transfer ops which the TT PJRT backend rejects at sync time with INTERNAL
Error code: 13.

Three transformers 5.x loader bugs were fixed before reaching this point:
1. `SlidingWindowCache` removed from `transformers.cache_utils` in 5.x — stub added
2. `use_fast=False` not passed to `AutoProcessor.from_pretrained` — Qwen2VLImageProcessor
   breaking change
3. `pad_token_id` not set on sub-configs (text_config) — AttributeError in model `__init__`
4. `'default'` key removed from `ROPE_INIT_FUNCTIONS` — model's custom RotaryEmbedding
   does a direct dict lookup

After those fixes the model loads and compiles but hits the D2H Error code: 13 inside
`rot_pos_emb` when `image_grid_thw` is on the TT device.

## Fix
**Loader fixes (committed to remediation branch):**
- `llava_onevision_1_5/pytorch/loader.py`: SlidingWindowCache stub, use_fast=False,
  pad_token_id on text_config, ROPE_INIT_FUNCTIONS 'default' shim

**Proposed fix for the Tier B bug:**
The TT PJRT backend needs to support device-to-host transfers for small integer tensors
inside compiled regions, OR the XLA Dynamo bridge needs to treat `tensor.cpu()` calls on
integer metadata (shape [N, 3] int64) as graph breaks that execute eagerly on the host.
This lives in `tt-xla`'s Dynamo bridge / PJRT execution path.

## Tier B justification
Indicator: `internal-error-unknown-mechanism` and `new-infrastructure`.

The INTERNAL Error code: 13 fires inside `torch_xla.sync()` during `extract_compiled_graph`
(the Dynamo compilation phase). Every approach that involves any D2H transfer inside the
compiled region — direct iteration yielding TT 0-dim tensors, explicit `.cpu()` calls, or
`.tolist()` — triggers the same error. Making this work requires either (a) PJRT support
for selective D2H transfers during execution, or (b) Dynamo graph-break support for
integer control-flow tensors. Neither is a scoped single-file change.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    295.63s (0:04:55, second run after loader fixes)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/llava_onevision_1_5/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 10fe56fa0b3a2286cb7e3e4fffe0eb3d3362169e |
