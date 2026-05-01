# Remediation Summary: lfm2/pytorch-LFM2_2_6B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2/pytorch-LFM2_2_6B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
lfm2-hybrid-conv-cache-non-tensor-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

## Root cause
`LiquidAI/LFM2-2.6B` has `config.use_cache=True` by default. When the model runs,
`Lfm2Model.forward()` creates an `Lfm2HybridConvCache` and returns it as
`past_key_values` in `CausalLMOutputWithPast`. Unlike standard KV caches,
`Lfm2HybridConvCache` does **not** inherit from `transformers.Cache`, so the
comparison evaluator's `_match_data_types` does not convert it. When `_compare_equal`
then calls `torch.utils._pytree.tree_map` over the outputs, the cache object is treated
as a non-container leaf and passed directly to `torch.equal()`, which rejects it.

## Fix
Added `model.config.use_cache = False` in `load_model` after loading the model.
This prevents `Lfm2HybridConvCache` from being created or returned — the model
produces only `logits` (and `past_key_values=None`) in its output, which the
comparison evaluator handles correctly.

File changed: `lfm2/pytorch/loader.py`

Commit: `fc83e80867ee0a0850aa7e8d49ed897e3dc1c391` on
`tenstorrent/tt-forge-models` branch
`remediation/lfm2-pytorch-LFM2_2_6B-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    57.45s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/lfm2/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | fc83e80867ee0a0850aa7e8d49ed897e3dc1c391 |
