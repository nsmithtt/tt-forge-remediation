# Remediation Summary: kronk3_5_9b_gguf-causal_lm-pytorch-9B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[kronk3_5_9b_gguf/causal_lm/pytorch-9B_GGUF-single_device-inference]

## Result
FAIL — after loader fix, SSM GatedDeltaNet scan loop (torch_chunk_gated_delta_rule) unrolls into ~1512 scatter ops in StableHLO; TT MLIR compiler hangs indefinitely

## Stack layer
loader, tt-xla, tt-mlir

## Tier
B

## Bug fingerprint
ssm-scan-loop-compilation-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise RuntimeError(
```
Two-phase failure. Before loader fix: `RuntimeError: You set ignore_mismatched_sizes to
False` — GGUF architecture "qwen35" was routed through Qwen3ForCausalLM (missing SSM
config fields → tensor shape mismatch). After loader fix: process hung for 57+ minutes
consuming 73 GB RAM during TT MLIR compilation of the SSM scan graph; killed before any
RuntimeError was raised.

## Root cause
Two issues encountered and fixed in sequence:

**Issue 1 (loader bug — fixed):**
1. No qwen35 architecture registration in the kronk3_5_9b_gguf loader.
2. A previously imported loader (mradermacher_qwen3_5_9b_abliterated_i1_gguf) installed
   a qwen35→qwen3 patch at import time that does not map SSM config fields (head_dim,
   num_key_value_heads, full_attention_interval etc.). Loading via qwen3 map with qwen35
   GGUF fields produces wrong tensor shapes → RuntimeError.
3. The import-time patch approach is not robust: any loader alphabetically later can
   overwrite the binding and break it (including loaders with missing `model_to_load`
   kwarg that causes a different TypeError in some session orders).
4. Missing requirements.txt with gguf>=0.10.0.

**Issue 2 (compiler-stack bug — Tier B, unfixed):**
After the loader fix, the model loads correctly as `Qwen3_5ForCausalLM` (32-layer hybrid
with 24 GatedDeltaNet SSM layers and 8 full-attention layers at every 4th block;
427/427 GGUF weights materialized). Because `flash-linear-attention` and `causal_conv1d`
are not installed, `Qwen3_5GatedDeltaNet.forward` falls back to `torch_chunk_gated_delta_rule`
— a pure-Python implementation with two Python for-loops:
  1. Inner triangular update: `for i in range(1, chunk_size)` — 63 iterations — each
     performing inplace slice assignments (`attn[..., i, :i] = ...`) that lower to
     scatter ops in StableHLO.
  2. Outer chunk loop: `for i in range(0, total_sequence_length // chunk_size)` — 2
     iterations for seq_len=128.

XLA/Dynamo unrolls these loops: 24 SSM layers × 63 inner iterations = 1512 scatter
operations materialized in the StableHLO graph. The TT MLIR compiler hangs when
optimizing this enormous scatter graph. Observed: RAM 22 GB → 73 GB over 57 minutes;
programs_log.yaml stopped updating at 08:20:39 (after initial trivial-op compilations);
process killed at ~09:09 with no forward progress.

## Fix
**Loader fix** — applied in 1 commit on
`remediation/kronk3_5_9b_gguf-causal_lm-pytorch-9B_GGUF-single_device-inference`
in tt_forge_models (commit 63cc69aacd):

1. Registers "qwen35" in all GGUF tables with full `_QWEN35_CONFIG_MAPPING` (includes
   SSM config fields: `linear_conv_kernel_dim`, `linear_value_head_dim`, `linear_num_key_heads`,
   `linear_num_value_heads`, `full_attention_interval`).
2. Adds `_Qwen35TensorProcessor` for correct tensor name mapping (GatedDeltaNet
   `linear_attn.*` ↔ `blk.N.ssm_*/attn_*` in GGUF).
3. Applies context manager (`_qwen35_gguf_context`) at call time (not import time) to
   override any previously installed bad patches from other loaders.
4. BFS over `__globals__` and `__closure__` cells to find the real transformers
   implementation (handles loaders that capture predecessor as closure variable).
5. Remaps `model_type` to `qwen3_5_text` when `full_attention_interval` is present
   in config (detects SSM hybrid, routes to `Qwen3_5ForCausalLM`).
6. Adds `requirements.txt` with `gguf>=0.10.0`.
7. Sets `use_cache=False` to prevent `Qwen3_5DynamicCache` (not a Cache subclass)
   from causing evaluator TypeError.

**Compiler-stack fix:** None attempted (Tier B).

**Proposed fix:** Implement a scan-loop recognition/fusion pass in the tt-mlir pipeline
that collapses repeated scatter patterns from `torch_chunk_gated_delta_rule` into a scan
primitive, or add a native scan primitive (linalg.scan or equivalent) to TTIR that
`Qwen3_5GatedDeltaNet` can be lowered into directly.

## Tier B justification
new-infrastructure: The fix requires either implementing a scan primitive in TTIR or adding
a loop-recognition/fusion pass spanning the tt-xla (tracing) and tt-mlir (compilation)
pipeline stages. This is new infrastructure across multiple compilation stages, not a
scoped single-file fix.

## Verification
- pytest exit: TIMEOUT
- Hardware:    blackhole-p150b
- Duration:    > 57 min (killed, no natural exit)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/kronk3_5_9b_gguf/causal_lm/pytorch/loader.py`
  (full loader rewrite with qwen35 GGUF support and context-manager patching)
- `tt-xla/third_party/tt_forge_models/kronk3_5_9b_gguf/causal_lm/pytorch/requirements.txt`
  (new: gguf>=0.10.0)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 63cc69aacd8425457c57e607af7b433e64199ac7 |
