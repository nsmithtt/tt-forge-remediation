# Remediation Summary: chatglm3-causal_lm-pytorch-6B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chatglm3/causal_lm/pytorch-6B-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-pretrainedconfig-max-length-removed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'. Did you mean: 'seq_length'?
```
Raised in `modeling_chatglm.py:863` during `ChatGLMForConditionalGeneration.__init__`:
```python
self.max_sequence_length = config.max_length
```

## Root cause
`transformers` 5.x removed `max_length` from `PretrainedConfig` (moved to `GenerationConfig`). The remote `modeling_chatglm.py` from `zai-org/chatglm3-6b` accesses `config.max_length` directly, which no longer exists. The loader also lacked a `use_cache` shim (transformers 5.x pops it from config init kwargs) and did not call `post_init()` (required by transformers 5.x to populate `all_tied_weights_keys`).

## Fix
Three changes in `chatglm3/causal_lm/pytorch/loader.py` in `tt-forge-models`:

1. Load `AutoConfig` first; if `max_length` is missing, set it from `seq_length`.
2. If `use_cache` is missing from config, default it to `True`.
3. Use `get_class_from_dynamic_module` (which sets the module hash) to patch `ChatGLMForConditionalGeneration.__init__` to call `post_init()` after the original init, so the patch survives the second module load inside `AutoModel.from_pretrained`.

Also removed the erroneous `torch_dtype` kwarg from the tokenizer call (tokenizers do not accept it).

File changed: `chatglm3/causal_lm/pytorch/loader.py`
Remediation branch: `remediation/chatglm3-causal_lm-pytorch-6B-single_device-inference` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    137.89s (0:02:17)
- Tier A attempts: N/A

## Files changed
- `chatglm3/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | f3ddbfb6b0eab2c2ec65fe45ec2347cc6ebedaca |
| tt-xla          | bf06ffb9565df110c35d5c50a6b7cc7a9d69b2c6 |
| tt-forge-models | abd5414d91c9ed86e254c0d2f3604d561f2e2b90 |
