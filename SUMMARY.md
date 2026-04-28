# Remediation Summary: aesthetic_shadow-pytorch-v2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aesthetic_shadow/pytorch-v2-single_device-inference]

## Result
FAIL — Conv2d patch embedding (kernel_size=64, out_channels=1536, output 16×16) exceeds L1 even with maximum DRAM auto-slice slicing; Tier B runtime bug in tt-metal

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
ttmetal-conv2d-dram-autoslice-small-spatial-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Two loader bugs were fixed before reaching the compiler-stack failure. Final failure after loader fixes:
```
TT_FATAL: DRAM Auto slice could not find valid slice configuration. Tried up to 16 slices for
height-slicing on output dimension 16. Available L1: 1396224 bytes. Operation requires more
memory than available even with maximum slicing.
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

### Loader bug 1: `spacy` namespace package shadowing

`dynamic_loader.py` adds `models_root` (the `tt_forge_models/` directory) to `sys.path` with the comment "so relative imports work". Because `tt_forge_models/spacy/` is a directory without `__init__.py`, Python creates a namespace package named `spacy` from it when `sys.path` is searched. Later, when `huspacy/pytorch/loader.py` does `import spacy` during model discovery, this namespace package is created and stored in `sys.modules['spacy']` — without a `Language` attribute. The HuggingFace `datasets` library's `_dill.py` checks `if "spacy" in sys.modules` and then calls `spacy.Language`, triggering `AttributeError: module 'spacy' has no attribute 'Language'`.

The `models_root` insertion into `sys.path` is not needed for relative imports (those work via `__package__` + the manually-registered `tt_forge_models` module). Removing it fixes the collision.

### Loader bug 2: Missing `preprocessor_config.json` for `RE-N-Y/aesthetic-shadow-v2`

The HuggingFace repo `RE-N-Y/aesthetic-shadow-v2` ships only `config.json` and `model.safetensors` — no `preprocessor_config.json`. `AutoImageProcessor.from_pretrained` fails with `OSError`. The loader creates a `ViTImageProcessor` directly from the model's `config.image_size` (1024) instead.

### Compiler-stack bug (unfixed): DRAM auto-slice L1 overflow for Conv2d patch embedding

The model's patch embedding is `Conv2d(in_channels=3, out_channels=1536, kernel_size=64, stride=64)` on a 1024×1024 input, producing a 16×16 output. In tt-metal's DRAM auto-slice path (`ttnn/operations/sliding_window/op_slicing/op_slicing.cpp`), the algorithm tries up to 16 height slices (one per output row) and 1 width slice (output_width=16 < TILE_HEIGHT=32, so only 1 width slice is possible). Even with 1 output row per slice, `calculate_L1_usage_for_conv_op` reports L1 usage that exceeds the available 1,396,224 bytes.

The large K dimension (3 × 64 × 64 = 12,288 weights per output pixel) and the degenerate output spatial dimensions (16×16 = 256 pixels, output_height < tile_height) likely produce a block configuration in `get_conv_configs` / `determine_parallel_config` that over-allocates L1. The exact mechanism requires deeper investigation of `calculate_L1_usage_for_conv_op`.

## Fix

### Loader bug 1 — `tt-xla` `tests/runner/utils/dynamic_loader.py`

Removed the `sys.path.insert(0, models_root)` call from `setup_models_path`. The manually-registered `tt_forge_models` module (with `__path__ = [models_root]`) handles all relative imports in loaders without needing `models_root` on `sys.path`.

Commit: `c87bc9bb2a23318d995ba71126ff09855f57430b` on branch `remediation/aesthetic_shadow-pytorch-v2-single_device-inference` in `tenstorrent/tt-xla`.

### Loader bug 2 — `tt_forge_models` `aesthetic_shadow/pytorch/loader.py`

Added `from transformers import ViTImageProcessor` and, after creating `VisionPreprocessor` and caching the model, injected a manually-constructed `ViTImageProcessor(size={"height": image_size, "width": image_size}, do_center_crop=False)` into `self._preprocessor._image_processor` to bypass the failed `AutoImageProcessor.from_pretrained`.

Commit: `e3f5ae0e99b899822517f32f92b8666875668bf5` on branch `remediation/aesthetic_shadow-pytorch-v2-single_device-inference` in `tenstorrent/tt-forge-models`.

### Proposed fix for compiler-stack bug

In `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`, investigate `calculate_L1_usage_for_conv_op` for the case where `output_height` < `TILE_HEIGHT` (16 < 32). The parallel config produced by `determine_parallel_config` for tiny outputs may assign degenerate shard shapes. A guard that detects this and falls back to a smaller-block or sequential (single-core) path might resolve the failure without requiring new infrastructure.

This is Tier B because:
- The root cause (why the L1 estimate exceeds available L1 even for 1-row slices) requires deep investigation of the `determine_parallel_config` → `get_conv_configs` → `calculate_L1_usage` chain.
- A correct fix would need to either: (a) adjust the parallel config for degenerate small-spatial outputs, or (b) use a reshape+matmul path for stride==kernel_size convolutions (new infrastructure).

## Tier B justification
internal-error-unknown-mechanism — The exact mechanism by which `calculate_L1_usage_for_conv_op` over-estimates L1 for a 1-row slice of a 16×16 output Conv2d is not yet determined. The correct fix requires investigation into the parallel/block config chain for sub-tile output spatial dimensions.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    100.83s (1:40) before runtime failure
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(0, models_root)`
- `tt_forge_models/aesthetic_shadow/pytorch/loader.py` — inject `ViTImageProcessor` from model config

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c87bc9bb2a23318d995ba71126ff09855f57430b |
| tt-forge-models | e3f5ae0e99b899822517f32f92b8666875668bf5 |
