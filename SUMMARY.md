# Remediation Summary: clinaligh-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[clinaligh/causal_lm/pytorch-30B_A3B_i1_GGUF-single_device-inference]

## Result
FAIL — device DRAM OOM during Qwen3Moe batched_mm expert dispatch; all three MoE implementations (eager/batched_mm/grouped_mm) incompatible with tt-xla

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
qwen3moe-moe-dispatch-tt-xla-incompatible

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `AttributeError: property 'gguf_file' of 'ModelLoader' object has no setter`

After fix 1: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After fix 2: `Fatal Python error: Segmentation fault` during `partition_fx_graph_for_cpu_fallback` (Qwen3Moe for-loop expert dispatch with `nonzero()`)

After fix 3: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`
Root cause: `TT_FATAL: Out of Memory: Not enough space to allocate 805306368 B DRAM buffer across 8 banks, where each bank needs to store 100663296 B, but bank size is 4273390016 B (allocated: 4263101568 B, free: 10288448 B)`

## Root cause

The loader had three sequential bugs all fixed:

**Bug 1 (loader):** `__init__` assigned `self.gguf_file = self._GGUF_FILES[self._variant]` but `gguf_file` was defined as a read-only property (no setter). The assignment is redundant since the property already computes the value.

**Bug 2 (loader):** 26 GGUF loaders monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0 now calls it with `model_to_load=dummy_model`, raising `TypeError`. Any of these 26 loaders imported during collection corrupts the global monkey-patch for all tests.

**Bug 3 (loader):** ClinAligh-30B-A3B is `qwen3_moe` architecture with 128 experts. The default `eager` expert dispatch uses `nonzero()` in a Python for-loop, which causes a segfault during `partition_fx_graph_for_cpu_fallback` in the XLA bridge. Fixed by setting `config._experts_implementation = "batched_mm"` before `from_pretrained`.

**Remaining compiler-stack bug (Tier B):** After applying `batched_mm`, the expert forward creates a `[S, 2*intermediate, hidden] = [1024, 1536, 2048]` intermediate tensor by gathering expert weights for all S=1024 selected token-expert pairs (128 tokens × 8 top-k). This allocation (768 MB = 96 MB per bank across 8 banks) fails because device DRAM is already 99.77% full (4.263 GB of 4.273 GB per bank). The three available implementations all fail with tt-xla:
- `eager`: segfaults XLA graph partitioner on `nonzero()` call
- `batched_mm`: OOMs DRAM on large intermediate tensor
- `grouped_mm`: uses `torch.histc` which fails on XLA (documented GLM-5 issue), and `torch._grouped_mm` has no known tt-mlir lowering

## Fix
**Loader fixes (committed):**
1. `clinaligh/causal_lm/pytorch/loader.py`: Remove redundant `self.gguf_file = ...` assignment in `__init__`.
2. 26 GGUF loader files: Change `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(*args, **kwargs):` and forward `*args, **kwargs` to the original.
3. `clinaligh/causal_lm/pytorch/loader.py`: Always load config and set `config._experts_implementation = "batched_mm"` before `from_pretrained`.

**Proposed compiler fix (Tier B):**
Add a tt-mlir/tt-xla lowering for `torch._grouped_mm` / `torch.nn.functional.grouped_mm` so that `grouped_mm` expert dispatch can run efficiently on device without materializing the full [S, 2*intermediate, hidden] intermediate tensor. This requires:
1. tt-mlir: implement a `ttir.grouped_mm` op with proper memory-efficient tiling
2. tt-xla: add StableHLO → TTIR lowering for `grouped_mm`
3. tt-metal: backend kernel for the grouped matmul operation

## Tier B justification
**Indicator:** `cross-cutting` — fixing requires new ops across tt-xla (StableHLO lowering), tt-mlir (new TTIR op), and tt-metal (kernel). Also the `new-infrastructure` indicator applies (implementing grouped_mm kernel is new infrastructure). Three repos, each needing coordinated changes.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1550.45s (0:25:50) — compiled successfully, OOM during first forward pass
- Tier A attempts: N/A

## Files changed
**tt_forge_models** (`remediation/clinaligh-causal_lm-pytorch-30B_A3B_i1_GGUF-single_device-inference`):
- `clinaligh/causal_lm/pytorch/loader.py` — fix property setter, add `_experts_implementation=batched_mm`
- 26 GGUF loader files — fix `_patched_load_gguf_checkpoint` narrow signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 65de68986f7e00ac545832ddf98d387ad72eb411 |
| tt-forge-models | b0f24bf7661df9e242cc0ab24d834b9febc3c1c9 |
