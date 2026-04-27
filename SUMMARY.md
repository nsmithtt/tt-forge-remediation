# AlinVLM v1.3 - HF Bringup 0

## Test

```
tests/runner/test_models.py::test_all_models_torch[alinvlm/pytorch-v1_3-single_device-inference]
```

## Original Failure

The CI run on branch `ip-172-31-30-236-tt-xla-dev/ubuntu/hf-bringup-0` reported:

```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Root cause: `Qwen3VLVisionModel.forward` calls `grid_thw.tolist()` (inside
`rot_pos_emb` / `fast_pos_embed_interpolate`) on tensors that land on the TT
XLA device.  On TT hardware, `.tolist()` triggers a synchronous XLA step that
fails with `Error code: 13 (INTERNAL)`.

## Diagnosis

AlinVLM v1.3 (`huiwon/alinvlm_v1_3`) is a **7B-class Qwen3VL model**:
- `text_config.hidden_size = 4096`
- `text_config.num_hidden_layers = 36`
- `text_config.intermediate_size = 12288`

Two distinct hardware limits are hit:

| Issue | Detail |
|-------|--------|
| Visual encoder | `Qwen3VLVisionPatchEmbed` uses `nn.Conv3d`; at native resolution the attention over 11,008 patches overflows TT L1. |
| Language model | The 7B text transformer overflows TT L1 by ~5× (8,204,800 B compiled vs 1,572,864 B max per core). |

For comparison, `qwen_3_vl/image_to_text/pytorch-2b_instruct` (2B model) is
already `KNOWN_FAILURE_XFAIL` for similar TT hardware limits.  A 7B model is
definitively beyond single-device TT capacity.

## Fixes Applied

### 1. Visual encoder patches in `alinvlm/pytorch/loader.py` (tt-forge-models)

Branch: `fix/alinvlm-v1_3-tt-hardware-patches`

Four patches applied at module load time via monkey-patching:

**a. `Qwen3VLVisionModel.forward` — run visual encoder on CPU**

```python
@torch.compiler.disable(recursive=True)
def _patched_visual_forward(self, hidden_states, grid_thw, **kwargs):
    param = next(self.parameters(), None)
    if param is not None and param.device.type != "cpu":
        self.cpu()   # move weights to CPU permanently on first call
    hidden_states = hidden_states.cpu()
    grid_thw = grid_thw.cpu()
    return _orig_visual_forward(self, hidden_states, grid_thw, **kwargs)
```

Prevents compilation of the visual encoder (Conv3d + 11k-patch attention);
runs it eagerly on CPU.  Also moves encoder parameters to CPU to avoid the
`Input type (CPUBFloat16Type) and weight type (XLABFloat16Type) should be the
same` error that arises when the enclosing model's weights are on XLA.

**b. `Qwen3VLModel.get_image_features` — CPU `image_grid_thw`**

Moves `image_grid_thw` to CPU before the `.prod(-1).tolist()` split-sizes
call inside `get_image_features`.

**c. `Qwen3VLModel.get_rope_index` — run rope computation on CPU**

```python
@torch.compiler.disable(recursive=True)
def _patched_get_rope_index(self, input_ids=None, image_grid_thw=None, ...):
    # move all inputs to CPU before the .tolist() control-flow calls
    ...
```

**d. `use_fast=False` + `max_pixels=512×512`**

`Qwen2VLImageProcessorFast` ignores `max_pixels`; the slow processor respects
it.  Capping at 512×512 gives ~988 patches instead of 11,008 (native
resolution), reducing visual encoder memory from ~2.1 MB to well within L1.

### 2. `KNOWN_FAILURE_XFAIL` entry in tt-xla test config

Branch: `fix/alinvlm-v1_3-xfail`

Added to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:

```yaml
alinvlm/pytorch-v1_3-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 —
    Statically allocated circular buffers grow to 8204800 B which is beyond
    max L1 size of 1572864 B.  AlinVLM v1.3 is a 7B-class Qwen3VL model
    (hidden_size=4096, 36 layers) that exceeds single-device TT L1 capacity."
```

With this entry the test exits `xfailed` (pytest exit code 0).

## Outcome

The test passes as `1 xfailed`.  The model cannot run on a single TT device
due to the language model L1 overflow.  The visual-encoder patches are retained
as they address real issues and will benefit smaller Qwen3VL-based models.

## Branches

- **tt-forge-models**: `fix/alinvlm-v1_3-tt-hardware-patches`
- **tt-xla**: `fix/alinvlm-v1_3-xfail`
