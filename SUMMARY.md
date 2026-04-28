# Remediation Summary: amlm_hard_nhot/masked_lm/pytorch-leukas/amlm_hard_nhot-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[amlm_hard_nhot/masked_lm/pytorch-leukas/amlm_hard_nhot-single_device-inference]

## Result
FAIL — LLVM ArrayRef out-of-bounds assertion in tt-mlir compilation pipeline (Tier B); three loader bugs fixed as prerequisite

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-arrayref-oob-assertion

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure (Python faulthandler crash output — the "Extension modules" list): a core dump during TT silicon compilation with the crash message:

```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253:
const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]:
Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
```

The crash occurs at `dynamo_bridge.py:483 in extract_graph_helper` → `_xla_warm_up_cache` during the TT compilation of the model's forward graph.

## Root cause

Three bugs were identified, in two layers:

**Bug 1 — loader (meta tensor):** Under transformers 5.x `init_empty_weights`, `NGramEmbeds.__init__` calls `self.ref_table = self.prepare_vocab_table()`. Because all tensor operations inside the meta-device context return meta tensors, `ref_table` becomes a meta tensor. It is NOT a registered parameter or buffer (the model code has `# self.register_buffer('ref_table', table)` commented out), so `from_pretrained` never replaces it with real weights. The forward pass then crashes at `self.ref_table.to(input_ids.device)` with `NotImplementedError: Cannot copy out of meta tensor; no data!`.

**Bug 2 — loader (dtype mismatch):** `prepare_vocab_table()` always creates a float32 eye matrix, while the model is loaded with `torch_dtype=bfloat16`. The `projection` linear layer has bfloat16 weights, causing `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16` at `self.projection(all_ids)`.

**Bug 3 — loader (device placement):** `ref_table` is a plain Python attribute, not a registered buffer. When `model.to(tt_device)` is called before the forward pass, registered parameters and buffers move to TT device, but `ref_table` stays on CPU. During the torch.compile forward pass, `self.ref_table.to(input_ids.device)` triggers a 3.2 GB (40000×40000 bfloat16) host-to-device copy inside the traced graph. The XLA warm-up cache then crashes with the LLVM assertion failure above.

**Bug 4 — tt-mlir (Tier B):** Even after the device placement is fixed so the `.to(device)` call is a no-op, the LLVM assertion `Index < Length` in `ArrayRef<long>::operator[]` fires during compilation of the model's forward graph. The Python faulthandler only provides the Python stack frame (`dynamo_bridge.py:483 → _xla_warm_up_cache`); the C++ backtrace (which would identify the specific MLIR pass) is not available. The model uses DeBERTa-v2 with custom NGram embeddings: the forward graph includes a `stablehlo.gather` from a 40000×40000 table, a linear layer over a 40000-dimensional space, and DeBERTa-v2 disentangled attention. The assertion failure requires C++ debugging (gdb/ASAN) to pinpoint which MLIR lowering pass is accessing an ArrayRef out of bounds.

## Fix

**Loader fixes** (in `tt_forge_models`, branch `remediation/amlm_hard_nhot-masked_lm-pytorch-leukas_amlm_hard_nhot-single_device-inference`):

File: `amlm_hard_nhot/masked_lm/pytorch/loader.py`

After `from_pretrained`, iterate over all modules and for any `NGramEmbeds` instance whose `ref_table` is still a meta tensor:
1. Call `prepare_vocab_table()` outside the meta-device context to get real weights.
2. Cast the table to the model's parameter dtype (fixing the bfloat16/float32 mismatch).
3. Pop the plain attribute from `module.__dict__` (so `register_buffer` does not reject it).
4. Call `module.register_buffer("ref_table", table, persistent=False)` so the buffer moves to TT device when `model.to(device)` is called, making the `.to(device)` call in forward a no-op.

**Compiler-stack fix** (proposed): In the tt-mlir compilation pipeline, the `ArrayRef<long>` out-of-bounds assertion must be diagnosed with a C++ debugger to identify which lowering pass accesses a shape array beyond its rank. Candidates are the `stablehlo.gather` lowering for the 40000×40000 table index, or one of the DeBERTa-v2 attention lowering passes. The fix would be a bounds check or correct indexing formula in the identified pass.

## Tier B justification (FAIL with Tier=B only)
`internal-error-unknown-mechanism` — The LLVM ArrayRef out-of-bounds assertion fires inside `_xla_warm_up_cache` (C++ code). The Python faulthandler provides only the Python call stack; the specific MLIR pass triggering the out-of-bounds access cannot be identified without a C++ stack trace from gdb or ASAN. Diagnosis must precede any fix.

## Verification
- pytest exit: FAIL (core dump / aborted)
- Hardware:    n150
- Duration:    ~65s (to compilation crash)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/amlm_hard_nhot/masked_lm/pytorch/loader.py` (loader, 3 commits)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c11707c43f07bca3979d0a897376eba2c3187ec2 |
