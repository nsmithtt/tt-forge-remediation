# Remediation Summary: bartowski_jackrong_qwen3_5_9b_neo_gguf-causal_lm-pytorch-Jackrong_Qwen3_5_9B_Neo_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_jackrong_qwen3_5_9b_neo_gguf/causal_lm/pytorch-Jackrong_Qwen3_5_9B_Neo_Q4_K_M-single_device-inference]

## Result
FAIL — GatedDeltaNet recurrent kernel produces PCC 0.768 on TT silicon vs CPU; root cause is float32 precision not preserved through TT lowering passes

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original (before loader fix):
`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

The bartowski_coniccat_qwen3_5_27b_writer_gguf loader (imported first alphabetically at
collection time) installs a `_patched_load_gguf_checkpoint` that doesn't accept the
`model_to_load` kwarg that transformers 5.x passes to `load_gguf_checkpoint`.

After loader fix:
`AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated:
pcc=0.7679169701241028. Required: pcc=0.99.`

## Root cause
Two separate issues were encountered and fixed in sequence:

**Issue 1 (loader bug — fixed):** Multiple issues with the GGUF loader:
1. The GGUF architecture string "qwen35" had no config-field or tensor-name mapping for
   `Qwen3_5ForCausalLM`. The coniccat loader (alphabetically earlier) mapped
   "qwen35" → "qwen3", loading the model as `Qwen3ForCausalLM` with wrong dimensions.
2. `_patched_load_gguf_checkpoint` in 25+ other Qwen3.5 loaders is missing `model_to_load=None`,
   meaning any of them being imported before this test runs will corrupt the global
   `load_gguf_checkpoint` binding. A context manager applied at call time (not import time)
   that BFS-walks the patch chain to find the real transformers function is the only robust fix.
3. Missing `requirements.txt` with `gguf>=0.10.0`.

**Issue 2 (compiler-stack bug — Tier B, unfixed):** After the loader fix, the model loads
correctly as `Qwen3_5ForCausalLM` (Qwen3.5-9B-Neo is a 28-layer SSM/full-attention hybrid
with 24 GatedDeltaNet linear-attention layers and 4 full-attention layers at every 4th block).
The model compiles on TT silicon without error, but produces PCC 0.768 instead of ≥0.99.

Root cause: `Qwen3_5GatedDeltaNet.forward` uses explicit `.float()` casts at numerically
sensitive points (e.g., `g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)`
in `modeling_qwen3_5.py`). TT does not preserve float32 through lowering, causing the
GatedDeltaNet recurrent states to be computed in bfloat16, introducing significant error.
This is the same bug as the 4B Neo variant (bartowski_jackrong_qwen3_5_4b_neo_gguf report,
bug fingerprint ttmlir-f32-precision-not-preserved, PCC 0.688).

## Fix
**Loader fix** — applied in 5 commits on
`remediation/bartowski_jackrong_qwen3_5_9b_neo_gguf-causal_lm-pytorch-Jackrong_Qwen3_5_9B_Neo_Q4_K_M-single_device-inference`
in tt_forge_models:

1. `06d8423dde` — Registers "qwen35" in all GGUF tables with full config-field mapping
   (`_QWEN35_CONFIG_MAPPING`), custom `_Qwen35TensorProcessor` for SSM weight transforms
   (A_log sign convention, conv1d reshape), and patches `get_gguf_hf_weights_map` to
   translate `qwen3_5_text → qwen35` for gguf-py arch lookup.

2. `1f55777e3b` — Switches from import-time patching to a `_qwen35_gguf_context()` context
   manager applied at call time so this loader's patch survives loaders imported after it
   alphabetically. Detects Neo via `full_attention_interval` key (survives the coniccat
   patch that already translates "qwen35" → "qwen3" at import time).

3. `5f2efa74c7` / `6faab61220` / `89f8e97f0d` — Progressive refinement of
   `_find_real_load_gguf_at_call_time()`: BFS over both `__globals__` (by known variable
   names) and `__closure__` cells to find the function whose `__globals__` is the
   `_gguf_utils` module dict — i.e., the real unpatched transformers implementation.
   Required because some loaders (e.g., momix_44) capture their predecessor as a closure
   variable rather than a module global.

Also adds `requirements.txt` (`gguf>=0.10.0`) and guards `apply_chat_template` with a
`chat_template is not None` check.

**Compiler-stack fix:** None attempted (Tier B).

**Proposed fix:** Ensure TT preserves float32 semantics for ops that follow an explicit
`.float()` cast in the StableHLO graph, or implement f32 accumulation paths in the TTNN
reduction and elementwise kernels used by GatedDeltaNet. This requires cross-cutting
changes to tt-mlir's lowering passes and is Tier B.

## Tier B justification
Indicator: **cross-cutting**

Preserving f32 precision through every lowering pass in tt-mlir is a cross-cutting change.
GatedDeltaNet uses explicit `.float()` at multiple points (gate computation, decay mask,
recurrent state casts) across 24 GatedDeltaNet layers. Fixing this requires ensuring f32
precision is maintained through elementwise, matmul, exp, and softplus ops in the TTNN
backend — changes spanning multiple lowering files and potentially the hardware kernel level.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    3946.20s (1:05:46)
- Tier A attempts: N/A (Tier B — no fix attempted in compiler stack)

## Files changed
- `tt-xla/third_party/tt_forge_models/bartowski_jackrong_qwen3_5_9b_neo_gguf/causal_lm/pytorch/loader.py`
  (full loader rewrite with qwen35 GGUF support and context-manager patching)
- `tt-xla/third_party/tt_forge_models/bartowski_jackrong_qwen3_5_9b_neo_gguf/causal_lm/pytorch/requirements.txt`
  (new: gguf>=0.10.0)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 641dbce4701bc9132b9624a72a6e6c509e221953 |
| tt-forge-models | 89f8e97f0df853efd52e0ccda50509d2752ad49f |
