# Remediation Summary: chatglm3-causal_lm-pytorch-6B-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chatglm3/causal_lm/pytorch-6B-Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-chatglm3-no-post-init-module-hash

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
(actual error: AttributeError: 'ChatGLMForConditionalGeneration' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?)

## Root cause
Three transformers 5.x incompatibilities in the ChatGLM3 remote model code
(loaded via trust_remote_code=True):

1. **max_length missing**: `ChatGLMConfig` stores the context length in
   `seq_length`, not `max_length`. transformers 5.x no longer auto-populates
   `max_length` on `PretrainedConfig`, so `ChatGLMForConditionalGeneration.__init__`
   raising `AttributeError: 'ChatGLMConfig' object has no attribute 'max_length'`.

2. **use_cache missing**: transformers 5.x pops `use_cache` from config init
   kwargs (it is now a generation parameter). The ChatGLM3 remote code accesses
   `config.use_cache` internally, causing `AttributeError`.

3. **all_tied_weights_keys missing**: `ChatGLMForConditionalGeneration.__init__`
   does not call `post_init()` (required in transformers 5.x to initialize
   `all_tied_weights_keys`). Patching `__init__` to call `post_init()` must use
   `get_class_from_dynamic_module` rather than `importlib.import_module`. The
   reason: `importlib.import_module` does not set `__transformers_module_hash__`
   on the module object, so the second call to `get_class_from_dynamic_module`
   inside `AutoModel.from_pretrained` re-executes the module and silently wipes
   the patch. Using `get_class_from_dynamic_module` first sets the hash, preventing
   re-execution and preserving the patched `__init__`.

## Fix
All three fixes are in `chatglm3/causal_lm/pytorch/loader.py` in the
`tt-forge-models` repo on branch
`remediation/chatglm3_causal_lm_pytorch-6B-Base-single_device-inference`:

- Set `config.max_length = config.seq_length` when `max_length` is absent.
- Set `config.use_cache = True` when `use_cache` is absent.
- Call `get_class_from_dynamic_module` to load and patch
  `ChatGLMForConditionalGeneration.__init__` so that it calls `post_init()`
  after the original `__init__` body runs (conditional on
  `all_tied_weights_keys` being absent). Switch `torch_dtype` → `dtype` for
  `from_pretrained` (deprecation in transformers 5.x).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    126.69s
- Tier A attempts: N/A

## Files changed
- `chatglm3/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7bfd0ca5ad0bcdbce79a5bf75ac62ebc6e63815b |
| tt-forge-models | 1078e838a809c778c4caf87e6fa2f8eb88749b07 |
