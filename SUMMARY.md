# Remediation Summary: darkidol_ballad_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[darkidol_ballad_gguf/causal_lm/pytorch-Q4_K_M-single_device-inference]

## Result
FAIL — MLIR compilation of 27B Qwen3.5 hybrid SSM model exceeds 30+ minutes without completing; two loader bugs fixed but compiler-stack compilation timeout prevents verification

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-qwen35-27b-hybrid-ssm-compilation-timeout

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fixes):
```
RuntimeError: You set 'ignore_mismatched_sizes' to 'False', thus raising an error.
```
Raised during `AutoModelForCausalLM.from_pretrained` when loading Darkidol-Ballad-27B-i1-Q4_K_M.gguf.
The GGUF architecture is `qwen35`; the shared monkey-patch in
`qwen_3_5_claude_distilled_gguf` mapped `ssm.time_step_rank` to `None` instead of
`linear_num_value_heads`, so the model was built with `linear_num_value_heads=32`
(the default) while the GGUF tensors encode 48 value heads. All SSM layer tensors
had mismatched shapes.

After fixing the config mapping:
```
AttributeError: 'Qwen3_5DecoderLayer' object has no attribute 'self_attn'
```
Raised in `darkidol_ballad_gguf/causal_lm/pytorch/loader.py:149` in `load_shard_spec`.
Qwen3.5 is a hybrid model where 3/4 of decoder layers use linear attention
(GatedDeltaNet SSM) with `linear_attn` rather than `self_attn`.

After both fixes:
```
[no error — compilation runs for 30+ minutes without output or completion]
```
The model loads correctly (851/851 tensors dequantized, fabric initialized on 4
devices), but MLIR compilation of the main 27B transformer body takes 30+ minutes
silently consuming CPU without producing log output or completing.

## Root cause
Three distinct root causes were found:

**Bug 1 (loader — fixed):** The `_patch_transformers_qwen35_gguf()` function in
`qwen_3_5_claude_distilled_gguf/causal_lm/pytorch/loader.py` maps GGUF metadata
fields to transformers config fields for the `qwen35` architecture. The entry for
`ssm.time_step_rank` was mapped to `None` (discarded), but for the 27B model this
field value is 48 and should map to `linear_num_value_heads`. The default fallback
of 32 caused all SSM layers to be constructed with wrong dimensions, triggering
`ignore_mismatched_sizes` errors on tensor loading.

**Bug 2 (loader — fixed):** `darkidol_ballad_gguf/causal_lm/pytorch/loader.py`
`load_shard_spec` iterated over decoder layers and unconditionally accessed
`layer.self_attn`. Qwen3.5 27B has 64 decoder layers, of which 48 use linear
attention (`layer.linear_attn`, no `self_attn` attribute) and 16 use full attention
(`layer.self_attn`). The missing guard caused `AttributeError` during shard spec
construction.

**Bug 3 (tt-mlir — Tier B, unfixed):** After both loader fixes the model loads and
TT Metal initializes successfully. MLIR compilation is attempted for the full 27B
model. Small auxiliary computations (attention mask generation, embedding lookup)
compile and execute within seconds. The main transformer forward-pass compilation
starts and runs for 30+ minutes (using 80–120% CPU on the host) without any log
output, error, or completion. The test was killed by timeout in every attempt:
- Run 1: killed after ~2 hours (background task timeout)
- Run 2: killed after <1 minute (residual memory pressure)
- Run 3: killed after 30 minutes (`timeout 1800`)

The tt-mlir verbose log (33K lines) shows the embedding subgraph pipeline completing
in under 5 seconds through vhlo → shlo → ttir → ttnn lowering. The main body
compilation begins at that point and produces no further output. The compilation is
actively consuming CPU (not hung/deadlocked) but the 64-layer 27B hybrid SSM body
is too large to compile within practical time bounds on this system.

## Fix
**Loader fix 1** — `tt-forge-models` repo, branch
`remediation/darkidol_ballad_gguf-causal_lm-pytorch-Q4_K_M-single_device-inference`:

File: `qwen_3_5_claude_distilled_gguf/causal_lm/pytorch/loader.py`

Changed the qwen35 GGUF config mapping in `_patch_transformers_qwen35_gguf()`:
```python
# Before:
"ssm.time_step_rank": None,
# After:
"ssm.time_step_rank": "linear_num_value_heads",
```
This causes `ssm.time_step_rank=48` from the 27B GGUF file to correctly set
`Qwen3_5TextConfig.linear_num_value_heads=48`, matching the shape of all SSM
tensors in the checkpoint.

Commit: `4aa9b15368`

**Loader fix 2** — same branch:

File: `darkidol_ballad_gguf/causal_lm/pytorch/loader.py`

Added `hasattr` guard in `load_shard_spec`:
```python
if hasattr(layer, "self_attn"):
    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
    ...
```
This allows `load_shard_spec` to handle hybrid Qwen3.5 layers that use
`linear_attn` instead of `self_attn`.

Commit: `33195a01b6`

**Proposed fix for Bug 3 (not attempted):** The MLIR compilation pipeline
requires cross-cutting improvements to compilation speed for large hybrid SSM
models. Possible approaches: AOT compilation result caching keyed by model
graph hash, parallel compilation of independent layer subgraphs, or a
compilation time limit with graceful fallback. This is outside the scope of a
single targeted Tier A fix.

## Tier B justification
cross-cutting: Reducing MLIR compilation time from 30+ minutes to a practical
bound for a 64-layer 27B hybrid SSM model would require cross-cutting changes
to the MLIR compilation pipeline (optimization passes, codegen, or caching
infrastructure) that span multiple files and subsystems.

## Verification
- pytest exit: TIMEOUT (killed after 30 min; model loaded successfully, compilation did not complete)
- Hardware:    wormhole (4 devices, n300)
- Duration:    >30 min (killed at timeout in all 3 attempts; main compilation never completed)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: qwen_3_5_claude_distilled_gguf/causal_lm/pytorch/loader.py` (ssm.time_step_rank mapping fix)
- `tt-forge-models: darkidol_ballad_gguf/causal_lm/pytorch/loader.py` (load_shard_spec hasattr guard)
- `tt-xla: third_party/tt_forge_models` (submodule pointer updated)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6db8ff9d84b6943f1ecf7f9fc2177ad55d7089c5 |
| tt-forge-models | 33195a01b6ae3903e12f9f04de2514e936de5e10 |
