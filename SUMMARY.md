# Remediation Summary: openelm-causal_lm-pytorch-270M_Instruct_mlx-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[openelm/causal_lm/pytorch-270M_Instruct_mlx-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
openelm-transformers5x-meta-device-persistent-false-buffers-uninit

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(Actual error captured during reproduction:)
TypeError: OpenELMForCausalLM.__init__() got an unexpected keyword argument 'use_cache'

## Root cause
Three loader bugs in `openelm/causal_lm/pytorch/loader.py`, all caused by the
`mlx-community/OpenELM-270M-Instruct` model using custom code via
`trust_remote_code=True` and transformers 5.x meta-device model initialisation:

1. **use_cache kwarg rejected**: `OpenELMForCausalLM.__init__` only accepts
   `config`; passing `use_cache=False` to `from_pretrained` causes a TypeError
   because transformers 5.x forwards unknown kwargs to `cls(config, **model_kwargs)`.

2. **Meta-device RoPE crash**: `OpenELMRotaryEmbedding.__init__` calls
   `_compute_sin_cos_embeddings` during construction.  Under
   `init_empty_weights()`, `inv_freq` is a meta tensor; the
   `emb.cos().to(device=cpu)` copy raises
   `NotImplementedError: Cannot copy out of meta tensor; no data!`.

3. **Uninitialised persistent=False buffers**: Both `inv_freq` (every
   `OpenELMRotaryEmbedding`) and `causal_mask` (`OpenELMModel`) are registered
   with `persistent=False`.  They are absent from the checkpoint, so after
   transformers 5.x materialises the model from meta device they contain
   uninitialised memory — giving garbage RoPE frequencies and a completely wrong
   attention mask, which caused PCC=nan then PCC=0.54 on silicon.

## Fix
All three fixes are in `tt_forge_models/openelm/causal_lm/pytorch/loader.py` on
branch `remediation/openelm-causal_lm-pytorch-270M_Instruct_mlx-single_device-inference`
of `tenstorrent/tt-forge-models`:

1. Load the config via `AutoConfig.from_pretrained`, set `config.use_cache = False`,
   and pass the pre-built `config` object to `from_pretrained` instead of the raw kwarg.

2. Use `get_class_from_dynamic_module` to load `OpenELMRotaryEmbedding` (which sets
   the module hash so `from_pretrained`'s second call does not re-execute the file and
   wipe the patch), then monkey-patch `_compute_sin_cos_embeddings` to be a no-op when
   `self.inv_freq.device.type == "meta"`.  The patch is restored in a `finally` block.

3. After `from_pretrained` returns, reinitialise:
   - `model.transformer.causal_mask`: reconstruct `torch.triu(torch.ones(ctx, ctx, bool), diagonal=1)`
   - Every `OpenELMRotaryEmbedding.inv_freq`: recompute `1.0 / (freq_constant ** (arange / model_dim))`
     and reset `_cached_cos`, `_cached_sin`, `_cached_seq_length`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    84.26s (0:01:24)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/openelm/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 04e8612f0411431ff08ac7cc5348d013f9c5d623 |
| tt-forge-models | 57d32c1af715f84d4359edb361b0d003f93f552a |
