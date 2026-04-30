# Remediation Summary: document_figure_classifier-pytorch-v2.0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[document_figure_classifier/pytorch-v2.0-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
avg-pool2d-ceil-mode-zero-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: shape '[1, 1280]' is invalid for input of size 0

While executing %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%avg_pool2d, [1, 1280]), kwargs = {})

## Root cause
`EfficientNetModel` uses `nn.AvgPool2d(config.hidden_dim, ceil_mode=True)` as its pooler (transformers `modeling_efficientnet.py:461`). For a 224×224 input, `hidden_dim=1280` and the final feature map is `[1, 1280, 7, 7]`. With `kernel_size=1280 >> input_size=7` and `ceil_mode=True`, PyTorch correctly computes output size 1×1 (the formula gives `ceil((7−1280)/1280)+1 = 1`). Without `ceil_mode=True` the formula gives `floor(…)+1 = 0`, producing an empty tensor.

XLA's `reduce_window` lowering for `aten.avg_pool2d` does not honour `ceil_mode=True`, so the output tensor gets shape `[1, 1280, 0, 0]` (size 0). The immediately following `aten.view.default` that reshapes to `[1, 1280]` then raises `RuntimeError: shape '[1, 1280]' is invalid for input of size 0`.

The bug surfaces in the experimental compile path inside `XLAExecutor._call_experimental_compile` (`backend.py:215`): `torch.export.export` produces an FX graph still containing the raw `aten.avg_pool2d.default` node (the custom decomposition in `decompositions.py` returned `NotImplemented` for this case), and `bridge.extract_compiled_graph` runs the graph on XLA tensors, triggering the shape mismatch.

## Fix
Extended the `avg_pool2d` custom decomposition in `tt-xla/python_package/tt_torch/backend/decompositions.py`.

When `ceil_mode=True` and, for every spatial dimension, `kernel > input_size` and `kernel − input_size < stride` (exactly one output position exists with `ceil_mode=True`, zero without it), the function now decomposes to `input.mean(dim=[-2, -1], keepdim=True)` instead of returning `NotImplemented`. With no explicit padding (`padding=0`), `count_include_pad` has no effect (the divisor equals the valid area, not the kernel area), so `mean()` is always the numerically correct reduction. A `divisor_override` is still respected via `sum / divisor_override`.

This replaces the problematic `aten.avg_pool2d.default` node in the FX graph before XLA sees it, allowing the model to compile and run correctly on TT hardware.

File changed: `tt-xla/python_package/tt_torch/backend/decompositions.py`
Branch: `remediation/document_figure_classifier-pytorch-v2.0-single_device-inference` in `tenstorrent/tt-xla`

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 59.81s
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/decompositions.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a84fa58a3fb78be52d68670b486ae56948009a0d |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
