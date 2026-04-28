# Remediation Summary: amd_qwen3_coder_next_mxfp4-causal_lm-pytorch-Qwen3_Coder_Next_MXFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[amd_qwen3_coder_next_mxfp4/causal_lm/pytorch-Qwen3_Coder_Next_MXFP4-single_device-inference]

## Result
FAIL — AMD Quark package unavailable; random-bfloat16 workaround produces 158 GB model that exceeds device DRAM and takes 30+ minutes to initialize

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
mxfp4-amd-quark-unavailable-model-exceeds-device-dram

## Workaround self-check
- Layer trimming: NO — removed the forbidden _DEFAULT_NUM_LAYERS=4 workaround
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
third_party/tt_forge_models/amd_qwen3_coder_next_mxfp4/causal_lm/pytorch/loader.py:161: in load_shard_spec
    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                ^^^^^^^^^^^^^^^
AttributeError: 'Qwen3NextDecoderLayer' object has no attribute 'self_attn'
```

## Root cause
Two issues:

**Issue 1 (fixed): Wrong attribute names in `load_shard_spec`.**
`Qwen3Next` is a hybrid architecture. Most layers (3 out of every 4) use
`linear_attn` (a `Qwen3NextGatedDeltaNet` module with projections
`in_proj_qkvz`, `in_proj_ba`, `out_proj`). Every fourth layer uses
`self_attn` (a `Qwen3NextAttention` module with the standard
`q_proj`/`k_proj`/`v_proj`/`o_proj` projections). The original
`load_shard_spec` assumed all layers had `self_attn`, causing
`AttributeError` on the 75% of layers that use `linear_attn` instead.

**Issue 2 (blocking): AMD Quark package unavailable.**
`amd/Qwen3-Coder-Next-MXFP4` requires AMD Quark to load its MXFP4
4-bit weights. Transformers raises `ImportError: Quark is not installed`
during `from_pretrained`. AMD Quark is not available via standard PyPI
(the `quark` PyPI package is unrelated OpenStack software; AMD Quark is
installed separately per https://quark.docs.amd.com/latest/install.html).

The branch worked around Issue 2 by stripping `quantization_config` and
using `from_config` with random bfloat16 weights + a forbidden
`_DEFAULT_NUM_LAYERS=4` layer trim. Removing the layer trim exposes the
real scale: the full 48-layer model with random bfloat16 weights is
~158 GB (48 layers × ~3.3 GB/layer, driven by 512-expert MoE blocks),
exceeding the p150b device DRAM. Even on CPU, initialization takes 30+
minutes and the test is not runnable in practice.

## Fix
**Applied to `tt-forge-models`, branch `remediation/amd_qwen3_coder_next_mxfp4-causal_lm-pytorch-Qwen3_Coder_Next_MXFP4-single_device-inference` (commit `8c5dcfaf67`):**

- Removed forbidden `_DEFAULT_NUM_LAYERS = 4` layer-trimming constant
  and the `self.num_layers = ... if ... else _DEFAULT_NUM_LAYERS` default.
- Replaced the incorrect `load_shard_spec` body (which assumed `self_attn`
  for all layers) with logic that dispatches on `hasattr(layer, "linear_attn")`:
  - For linear-attention layers: shard `linear_attn.in_proj_qkvz.weight`,
    `linear_attn.in_proj_ba.weight`, and `linear_attn.out_proj.weight`.
  - For full-attention layers: shard `self_attn.q_proj.weight`,
    `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, and
    `self_attn.o_proj.weight`.

**Remaining (unresolved): AMD Quark dependency.**
The test cannot proceed to silicon without either (a) installing AMD Quark
so `from_pretrained` loads the native 4-bit weights (~40 GB, fits on p150b),
or (b) a supported way to load the model without quark that does not inflate
the weight size to 158 GB. Adding `amd-quark` (or its correct package name)
to `requirements.txt` is the proposed path forward.

## Verification
- pytest exit: FAIL (killed during from_config after 6+ minutes; never reached silicon)
- Hardware:    p150b (Blackhole, ~144 GB device DRAM)
- Duration:    not-run (6+ min CPU initialization, model never reached device)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/amd_qwen3_coder_next_mxfp4/causal_lm/pytorch/loader.py`
  (remediation branch: `8c5dcfaf67`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 8c5dcfaf677458ac1993fb08faf888a2c9bc4ada |
