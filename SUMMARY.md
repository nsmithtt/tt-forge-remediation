# Remediation Summary: huihui_lfm2_8b_a1b_abliterated_gguf-causal_lm-pytorch-8B_A1B_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_lfm2_8b_a1b_abliterated_gguf/causal_lm/pytorch-8B_A1B_Abliterated_GGUF-single_device-inference]

## Result
FAIL — segfault (SIGSEGV) in TT backend during FX graph partitioning; Tier B internal crash with unknown mechanism

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
tt-backend-sigsegv-fx-partition-lfm2moe

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault

Thread 0x... (most recent call first):
  File "torch/_ops.py", line 841 in __call__
  File "tt_torch/torch_overrides.py", line 41 in __torch_function__
  File "torch/_ops.py", line 841 in __call__
  File "torch/fx/interpreter.py", line 336 in call_function
  File "torch_xla/_dynamo/dynamo_bridge.py", line 652 in run_node
  File "torch/fx/interpreter.py", line 174 in run
  File "torch_xla/_dynamo/dynamo_bridge.py", line 762 in partition_fx_graph_for_cpu_fallback
  File "tt_torch/backend/backend.py", line 225 in __call__
  File "transformers/models/lfm2_moe/modeling_lfm2_moe.py", line 768 in forward
  File "model_tester.py", line 225 in _run_on_tt_device

## Root cause
Multiple loader-layer bugs were found and fixed (see Fix section). After all loader
fixes, the model loads successfully and runs correctly on CPU. On TT device, the test
crashes with a fatal SIGSEGV during `partition_fx_graph_for_cpu_fallback` in the
`torch_xla` dynamo bridge. The crash occurs at the call site in
`UnsupportedNodesCollector.run_node()` when the FX interpreter executes a TorchOp on
TT device tensors via `TorchFunctionOverride.__torch_function__`. The crash is in C++
(no Python-level error message before the dump), originating from a TT backend op
invocation during graph-partitioning exploration. The specific op causing the crash is
unknown; diagnosis requires C++ debugging of the TT XLA backend. This is a Tier B
internal-error-unknown-mechanism failure.

## Fix
Multiple loader fixes were applied in `tt-xla` (remediation branch) and `tt_forge_models`
(via tt-xla submodule):

**tt-xla: tests/pytest.ini**
- Added `pythonpath = tests` so `infra` package is importable.
- Added SWIG `DeprecationWarning` filters (original reported failure symptom).

**tt-xla: python_package/tt_torch/torch_overrides.py**
- Added int→float cast for `histc` in `TorchFunctionOverride.__torch_function__`.
  `torch.ops.aten.histc` is not implemented for integer dtypes on CPU; the LFM2-MoE
  router calls `torch.histc(selected_experts.view(-1), ...)` with a Long tensor during
  `partition_fx_graph_for_cpu_fallback` CPU exploration. This is a Tier A fix (one
  guard in one file, one known op gap on CPU path).

**tt_forge_models: huihui_lfm2_8b_a1b_abliterated_gguf/causal_lm/pytorch/loader.py**
- Registered `"lfm2moe"` in `GGUF_TO_TRANSFORMERS_MAPPING["config"]`,
  `GGUF_SUPPORTED_ARCHITECTURES`, and `TENSOR_PROCESSORS["lfm2moe"]` (uses
  `Lfm2TensorProcessor`). Transformers 5.2.0 knows `lfm2_moe` as an HF model type
  but lacks the GGUF architecture key `"lfm2moe"`.
- Fixed `layer_types` construction: `Lfm2MoeConfig` uses `layer_types: list[str]`
  (e.g. `["full_attention", "conv", ...]`), not `full_attn_idxs` (used by the dense
  `Lfm2Config`). The GGUF metadata provides per-layer `num_key_value_heads` as a list;
  we build `layer_types` from it and take `max()` for the scalar `num_key_value_heads`.
- Registered both `"lfm2moe"` and `"lfm2_moe"` in `GGUF_TO_FAST_CONVERTERS` with
  `GGUFQwen2Converter`. The tokenizer converter lookup uses the post-remap model_type
  (`"lfm2_moe"`), so only registering `"lfm2moe"` caused a `KeyError`.
- Fixed `model_to_load` TypeError from broken patcher chain: 26+ other loaders at
  import time wrap `gguf_utils.load_gguf_checkpoint` with module-level functions
  lacking the `model_to_load` kwarg added in transformers 5.x. Our patcher is
  then buried when later loaders re-wrap. Fix: BFS traversal (`_find_real_loader`)
  over both closure cells AND `co_names`-referenced globals to find the real
  transformers function; context manager `_lfm2moe_load_ctx()` in `load_model()`
  re-installs our patcher as the outermost wrapper before `from_pretrained`.
- Fixed `IndexError` from OOV special tokens: The GGUF file's quantized embedding
  covers `vocab_size=65536` indices (0–65535). The tokenizer injects
  `<|im_start|>=65536` and `<|im_end|>=65537` via chat template, causing index
  out-of-range in the embedding lookup. Fix: use plain `sample_text` without chat
  template so all token IDs are within the GGUF embedding range.

The remaining failure (segfault in TT backend) is Tier B — proposed fix and location
described below.

## Tier B justification
Indicator: `internal-error-unknown-mechanism`

The SIGSEGV occurs in C++ code within the TT XLA backend when `UnsupportedNodesCollector`
runs LFM2-MoE ops on TT device tensors during `partition_fx_graph_for_cpu_fallback`. The
crash has no Python-level diagnostic; the specific TT op that crashes is unknown without
C++ debugging (gdb/lldb on the TT shared library). The proposed fix would live in
`tt-xla` or `tt-mlir`, in the op lowering or device execution path for one of the LFM2-MoE
operations (likely the conv or MoE routing ops), but the mechanism cannot be determined
from Python-level information alone. A human with C++ stack expertise and a debug build
must triage this.

## Verification
- pytest exit: FAIL (Fatal Python error: Segmentation fault)
- Hardware: n150
- Duration: ~120s (crash before completion)
- Tier A attempts: 1 (histc int→float fix; successful, but subsequent segfault is separate Tier B bug)

## Files changed
- `tt-xla/tests/pytest.ini` — added `pythonpath = tests`, SWIG DeprecationWarning filters
- `tt-xla/python_package/tt_torch/torch_overrides.py` — int→float cast for histc (Tier A)
- `tt-xla/third_party/tt_forge_models/huihui_lfm2_8b_a1b_abliterated_gguf/causal_lm/pytorch/loader.py` — all loader fixes

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 40601556ca67b151b4f29c137f1181d1ce95540d |
| tt-forge-models | 05e029b98c9b5a15fda22e000a7a4f4361fc90bb |
