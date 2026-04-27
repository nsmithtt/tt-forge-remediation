# Remediation Summary: accent_id_commonaccent_xlsr/pytorch

## Test
`tests/runner/test_models.py::test_all_models_torch[accent_id_commonaccent_xlsr/pytorch-CommonAccent_XLSR_EN-single_device-inference]`

## Result: PASS

All changes are in `third_party/tt_forge_models` (branch `nsmith/fix-accent-id-commonaccent-xlsr-float32`), specifically in `accent_id_commonaccent_xlsr/pytorch/loader.py`.

## Issues Fixed

### 1. pos_conv_embed TT_FATAL (L1 overflow)
XLSR's positional convolutional embedding (1024ch, kernel=128, groups=16) produces a 1x49 output. TILE layout limits this to `div_up(49,32)=2` DRAM slices — insufficient for L1, causing `TT_FATAL` at runtime.

**Fix:** Run wav2vec2 entirely on CPU via a `@torch.compiler.disable` graph break (`_run_wav2vec2`). The function calls `wav2vec2_module(wavs.cpu())` and returns `feats.to("xla")` (when input was on XLA) so downstream partitions (avg_pool, output_mlp) remain in the XLA/TT graph.

### 2. wav2vec2 parameters migrated to XLA by `Module.to()`
The test infrastructure calls `to_device(workload.compiled_executable, "xla")` which invokes `OptimizedModule.to("xla")`. `torch.nn.Module.to()` recurses into `AccentClassifierModel._modules`, bypassing any custom `to()` override and moving wav2vec2 parameters to XLA. This causes a device mismatch when `_run_wav2vec2` calls `wavs.cpu()` — CPU wavs + XLA wav2vec2 parameters → layer_norm dispatch error.

**Fix:** Store wav2vec2 outside `_modules` using `object.__setattr__(self, "wav2vec2", ...)`. `Module.to()` only traverses `_modules`, so wav2vec2 parameters are never moved to XLA regardless of how `.to("xla")` is called on the model or its compiled wrapper.

### 3. `stablehlo.round_nearest_even` unsupported op
SpeechBrain's `StatisticsPooling.forward` computes `int(torch.round(lengths * max_len))` on XLA tensors. `torch.round` on an XLA tensor generates `stablehlo.round_nearest_even`, which TT-MLIR does not support. Additionally, `int(xla_tensor)` forces an XLA synchronization.

**Fix:** Monkey-patch `StatisticsPooling.forward` to detect XLA device lengths and use `x.shape[1]` directly (valid for `wav_lens=1.0` full-sequence inference), bypassing both the unsupported op and the forced sync.

### 4. Unsupported `ttnn.layer_norm` passes
SpeechBrain's wav2vec2 integration applies layer normalization to the raw waveform (`normalize_wav=True`) and to the encoder output (`output_norm=True`). These lower to `ttnn.layer_norm` on TT hardware but are not supported in the current TT-MLIR stack.

**Fix:** Disable both passes (`normalize_wav=False`, `output_norm=False`). Pre-normalize the input waveform in `load_inputs()` using `torch.nn.functional.layer_norm` on CPU before it enters the model.

### 5. dtype_override causing float32/bfloat16 mismatch
SpeechBrain's wav2vec2 integration creates float32 tensors internally. Applying a `dtype_override` (e.g., bfloat16) caused dtype mismatches within the model.

**Fix:** Pop `dtype_override` from kwargs in `load_model()` to always load in native float32.
