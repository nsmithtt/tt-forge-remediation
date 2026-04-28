# Remediation Summary: llava_onevision_1_5-pytorch-4B_Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision_1_5/pytorch-4B_Base-single_device-inference]

## Result
FAIL — torch.arange with TT device tensor fails inside compiled XLA graph (pjrt-device-to-host-transfer)

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

Traceback (innermost relevant frames):
  modeling_llavaonevision1_5.py:1023 — rotary_pos_emb = self.rot_pos_emb(grid_thw)
  modeling_llavaonevision1_5.py:949  — hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
  tt_torch/torch_overrides.py:34     — return func(*args, **(kwargs or {}))
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Three loader-layer bugs blocked the model from loading under transformers 5.x; all three were fixed:

1. **SlidingWindowCache removed** — `transformers.cache_utils` no longer exports `SlidingWindowCache` in 5.x. The model's trust_remote_code module imports it at the top level for `isinstance` checks in `_update_causal_mask`.

2. **pad_token_id absent from sub-config** — In transformers 5.x, generation-related attributes (including `pad_token_id`) are no longer automatically set on `PretrainedConfig` sub-configs. `LLaVAOneVision1_5_TextModel.__init__` accesses `config.pad_token_id` to set the embedding `padding_idx`; it raises `AttributeError` when the attribute is absent.

3. **'default' missing from ROPE_INIT_FUNCTIONS** — `transformers 5.x` removed the `'default'` key from `ROPE_INIT_FUNCTIONS`; the standard RoPE is now built into the base `RotaryEmbedding` class. The model's custom `LLaVAOneVision1_5_RotaryEmbedding` still does `ROPE_INIT_FUNCTIONS[self.rope_type]` with `rope_type='default'`, causing a `KeyError`.

After fixing all three loader bugs, the model loads and compilation begins. The test then fails with `INTERNAL: Error code: 13` in the visual encoder's `rot_pos_emb` function:

```python
# modeling_llavaonevision1_5.py:947-949
for t, h, w in grid_thw:        # h, w are 0-dim TT device tensors
    hpos_ids = torch.arange(h)   # torch.arange with a device tensor
```

During `torch.compile` execution, `grid_thw` is a TT device tensor. Iterating over it yields 0-dim TT tensors `h` and `w`. Calling `torch.arange(h)` with a TT device tensor requires the PJRT backend to transfer the scalar value device→host to determine the output size. The TT PJRT backend does not support this device-to-host transfer of integer scalars inside compiled XLA graphs, resulting in `INTERNAL: Error code: 13`.

## Fix
**Loader layer (committed, pushed):**
Three commits on `remediation/llava_onevision_1_5-pytorch-4B_Base-single_device-inference` in `tt-forge-models`:

- `70e7ddf` — Add `SlidingWindowCache` stub to `transformers.cache_utils` and pass `use_fast=False` to `AutoProcessor.from_pretrained`
- `96ec902` — Pre-load config, set `config.text_config.pad_token_id = None` if absent, then pass the patched config to `AutoModelForCausalLM.from_pretrained`
- `cee8fbd` — Add `'default'` key to `ROPE_INIT_FUNCTIONS` using the standard RoPE inverse-frequency formula

**Compiler-stack layer (Tier B — not attempted):**
The fix would live in `tt-xla` (PJRT backend). When lowering `torch.arange(dynamic_size)` or `aten.arange.start_step` with a device tensor as the stop argument, the XLA/StableHLO path needs to either:
- Emit a `stablehlo.dynamic_iota` with the tensor value correctly transferred, or
- Route through a `stablehlo.iota` after binding the scalar via a PJRT device-to-host read.

This requires adding a new lowering path for dynamic `arange` in the PJRT bridge, which is new infrastructure (Tier B indicator: `new-infrastructure`).

## Tier B justification
Which indicator: `new-infrastructure`

Supporting `torch.arange` with a dynamic device tensor as the size argument requires either implementing `stablehlo.dynamic_iota` lowering in the TT compiler pipeline, or implementing device-to-host scalar transfer for compile-time constant folding in the PJRT bridge. Neither path exists today; both are new infrastructure additions.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    236.98s (0:03:56) to reach the compiler-stack failure after loader fixes
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/llava_onevision_1_5/pytorch/loader.py` (3 commits on remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | cee8fbd426b373a5e25681091522d4b0c8fc3f08 |
