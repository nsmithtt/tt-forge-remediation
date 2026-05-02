# Remediation Summary: qwen_14b-causal_lm-pytorch-Qwen-14B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_14b/causal_lm/pytorch-Qwen-14B-single_device-inference]

## Result
SILICON_PASS — 4 loader bugs fixed; PCC=0.9912 on n150 BF16 (threshold 0.99)

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: This modeling file requires the following packages that were not found in your environment: tiktoken. Run `pip install tiktoken`
```

## Root cause
Four loader bugs:

1. **Missing tiktoken dependency**: Qwen/Qwen-14B uses a custom TikToken-based tokenizer (`QWenTokenizer`). The `check_imports` function in transformers scans all imports in the model file (including those inside function bodies) and raises if any package is missing. `tiktoken` was not declared in `requirements.txt`.

2. **transformers_stream_generator incompatible with transformers 5.x**: After tiktoken was added, `check_imports` found the lazy `from transformers_stream_generator.main import ...` inside `chat_stream()`. Installing `transformers_stream_generator==0.0.5` fails on import because it references `DisjunctiveConstraint` which was removed in transformers 5.x. This package is only used by `chat_stream()` which is never called during inference, so the fix is to inject a stub module into `sys.modules` before the model loads.

3. **QWenTokenizer pad_token is None**: The `QWenTokenizer` TikToken backend does not populate the standard transformers `eos_token` field (returns `None`). The original code set `pad_token = eos_token` which left it as `None`, causing `ValueError: Asking to pad but the tokenizer does not have a padding token` in `load_inputs`. Fix: set `pad_token = '<|endoftext|>'` directly.

4. **RotaryEmbedding cos/sin cache device mismatch**: `RotaryEmbedding` stores `_rotary_pos_emb_cache` as a plain Python list attribute (not a registered buffer). When the model is run on CPU first (golden reference), the cache is built with CPU tensors. When the model is subsequently moved to XLA via `.to(device)`, the registered buffer `inv_freq` moves to XLA but the plain list `_rotary_pos_emb_cache` stays on CPU. The cache rebuild guard (`seqlen > _seq_len_cached`) is not triggered by device change, so the stale CPU tensors are used when the XLA-placed model forward is traced by Dynamo, causing `RuntimeError: Unhandled FakeTensor Device Propagation for aten.mul.Tensor, found two different devices xla:0, cpu`.

## Fix
All changes are in `tt-xla/third_party/tt_forge_models/qwen_14b/causal_lm/pytorch/`:

1. Added `requirements.txt` with `tiktoken`.

2. In `loader.py`: at module level (before the model is loaded), inject a stub for `transformers_stream_generator` and `transformers_stream_generator.main` into `sys.modules` so `check_imports` finds and successfully imports the package without executing the broken real package code.

3. In `loader.py` `_load_tokenizer`: set `self.tokenizer.pad_token = "<|endoftext|>"` instead of `self.tokenizer.eos_token` (which is `None` for this tokenizer).

4. In `loader.py` `_patch_rotary_embedding`: after `from_pretrained`, iterate all `RotaryEmbedding` modules and wrap `update_rotary_pos_emb_cache` to first check if `inv_freq.device != _rotary_pos_emb_cache[0].device`; if so, reset `_seq_len_cached = 0` to force a cache rebuild on the current device before returning cos/sin.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    ~277s (4:37) wall-clock
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/qwen_14b/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_14b/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1f37de6e2bba4c42fb4c4d61b9e1238da8ea544c |
| tt-forge-models | 52473a37062be3cd84c3549436121aee5d2409ec |
