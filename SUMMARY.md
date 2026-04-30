# Remediation Summary: cosmos_embed1-feature_extraction-pytorch-Cosmos-Embed1-448p-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cosmos_embed1/feature_extraction/pytorch-Cosmos-Embed1-448p-single_device-inference]

## Result
FAIL — SIGABRT during second tt::runtime::submit call (trivial constants graph, second run after large model forward pass); no TT_FATAL message visible; Tier B runtime bug

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
ttmetal-runtime-sigabrt-second-submit-after-large-forward

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Aborted (exit code 134)

Full sequence:
1. Model loads and compiles successfully (three loader bugs fixed — see Fix section).
2. First TT device run: SyncTensorsGraph.4 (two bf16 scalar constants) compiles and executes successfully. Main model graph (SyncTensorsGraph.21872) compiles and executes successfully.
3. Second TT device run (warmup #1 of _test_e2e_perf): SyncTensorsGraph.4 re-executed from cache. Process aborts with SIGABRT inside tt::runtime::submit, with no preceding TT_FATAL or TT_THROW log message.

Last log lines before abort:
```
FlatbufferLoadedExecutableInstance::Execute
ClientInstance::getOrCreateMeshDevice - reusing already opened mesh device [1, 1]
[concurrent thread] PJRT_Client_BufferFromHostBuffer (x2)
Fatal Python error: Aborted
```

## Root cause
Two distinct bug layers:

**Loader layer (fixed):** Three transformers 5.x breaking changes broke model loading:
1. `RuntimeError: Tensor.item() cannot be called on meta tensors` — `EvaViTG.__init__` calls `torch.linspace(...).item()` inside the meta-device init context that transformers 5.x uses.
2. `RuntimeError: You set ignore_mismatched_sizes to False` — transformers 5.x skips `_load_state_dict_pre_hooks` during `from_pretrained`, so `PositionalEmbeddingHook` (which interpolates `visual_encoder.pos_embed` from [1,257,1408] to [1,1025,1408]) never runs.
3. `AttributeError: 'CosmosEmbed1' object has no attribute 'all_tied_weights_keys'` — `CosmosEmbed1.__init__` does not call `post_init()`, so `all_tied_weights_keys` is never set; `_finalize_model_loading` accesses it unconditionally.

**Runtime layer (unfixed, Tier B):** After the three loader fixes are applied, the model loads and compiles. However, on the second call to tt::runtime::submit for the trivial constants graph (logit_scale/logit_bias only), the process aborts with SIGABRT. No TT_FATAL or TT_THROW message appears in the log, ruling out the normal exception path. The abort follows the main model forward pass (SyncTensorsGraph.21872, ~70k MLIR lines) without any error from that pass. Root cause in tt-metal is unknown — the abort is silent and likely occurs in a dispatch worker thread or device memory subsystem.

## Fix
**Loader fixes (committed) in `tt_forge_models` on branch `remediation/cosmos_embed1-feature_extraction-pytorch-Cosmos-Embed1-448p-single_device-inference` (commit 43e275901d):**

`cosmos_embed1/feature_extraction/pytorch/loader.py`:
- Added `ignore_mismatched_sizes=True` to `model_kwargs`
- Patched `torch.Tensor.item` to return `0.0` for meta scalars during `from_pretrained`
- After loading, manually retrieve pos_embed shard via safetensors and apply interpolation via `model.visual_encoder.load_state_dict({"pos_embed": pos_embed_ckpt}, strict=False)` — PyTorch's `load_state_dict` calls `_load_state_dict_pre_hooks`, triggering `PositionalEmbeddingHook`

`cosmos_embed1/feature_extraction/pytorch/src/model_utils.py`:
- Added `_patched_adjust_tied` that initializes `all_tied_weights_keys = {}` if absent before delegating to the original `_adjust_tied_keys_with_tied_pointers`

**Runtime fix (proposed, not attempted):** Investigate why `tt::runtime::submit` aborts silently on its second invocation for a constants-only program after a large model forward pass. The dispatch worker thread likely encounters a resource use-after-free or state corruption. Look at program/tensor pool cleanup in `ProgramExecutor::execute()` and dispatch queue flushing between programs.

## Tier B justification
- The abort is SIGABRT with no TT_FATAL/TT_THROW log — the root cause is an INTERNAL error without a known mechanism (indicator: `internal-error-unknown-mechanism`)
- The failure involves the tt-metal dispatch/runtime subsystem and device memory management post large-model-forward; fixing it requires deep investigation of the dispatch worker threading model, which is not a scoped single-function change

## Verification
- pytest exit: FAIL (exit code 134, SIGABRT)
- Hardware:    n150
- Duration:    ~56s before abort (compilation ~12s constants + ~12s for main model)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/cosmos_embed1/feature_extraction/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/cosmos_embed1/feature_extraction/pytorch/src/model_utils.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 43e275901d101e75fb8a3bc800819fe27d4a7f8b |
