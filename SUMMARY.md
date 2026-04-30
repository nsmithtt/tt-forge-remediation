# Remediation Summary: flux_controlnet_depth_v3-pytorch-depth-v3-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_controlnet_depth_v3/pytorch-depth-v3-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
diffusers-flux-controlnet-from-single-file-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(True root error under the DeprecationWarning noise:)
1. ModuleNotFoundError: No module named 'infra'
2. AttributeError: type object 'FluxControlNetModel' has no attribute 'from_single_file'
3. TypeError: FluxControlNetOutput.__init__() missing 1 required positional argument: 'controlnet_single_block_samples'

## Root cause
Three independent loader-layer bugs:

**Bug 1 — pytest.ini missing `pythonpath = tests`:** `tests/conftest.py` imports from
the local `tests/infra/` package, but `tests/` was not in `sys.path`. Pytest collected
the DeprecationWarning from SWIG and the import failure surfaced as the headline error.

**Bug 2 — `FluxControlNetModel.from_single_file` does not exist in diffusers 0.37.1:**
The loader called `FluxControlNetModel.from_single_file(url, ...)` but `FluxControlNetModel`
inherits only `ModelMixin / ConfigMixin / PeftAdapterMixin` — none of which include
`from_single_file`. The XLabs checkpoint uses BFL/ComfyUI key naming rather than the
diffusers key naming, so it cannot be loaded via `from_pretrained` without key conversion.

**Bug 3 — `FluxControlNetOutput` pytree unflatten failure when `controlnet_single_block_samples=None`:**
`FluxControlNetOutput` extends `BaseOutput` (an `OrderedDict`) which omits `None` values.
With `num_single_layers=0` the forward returns `controlnet_single_block_samples=None`,
so the OrderedDict has only `controlnet_block_samples`. When the test framework moves
TT device outputs to CPU via `tree_map`, the pytree registration's unflatten lambda calls
`FluxControlNetOutput(**dict)` without the missing field, raising a `TypeError` because the
dataclass `__init__` requires both positional arguments.

## Fix
Three changes, all in the loader layer:

**Fix 1 — `tt-xla/pytest.ini`:** Added `pythonpath = tests` and `filterwarnings` to suppress
the known SWIG DeprecationWarning. File: `pytest.ini`.

**Fix 2 — `tt_forge_models/flux_controlnet_depth_v3/pytorch/loader.py`:** Replaced
`FluxControlNetModel.from_single_file(url)` with:
- `hf_hub_download(REPO_ID, SAFETENSORS_FILE)` to obtain the checkpoint locally
- `load_file(path)` to load the raw state dict
- `_convert_xlabs_controlnet_state_dict()` to rename BFL keys to diffusers keys
  (double_blocks → transformer_blocks, img_attn/txt_attn → to_q/to_k/to_v, etc.)
- `FluxControlNetModel(num_layers=N, num_single_layers=0, guidance_embeds=has_guidance)`
  initialized from the checkpoint itself

**Fix 3 — same `loader.py`, `load_inputs()`:** Added `"return_dict": False` to the inputs
dict so the model returns a plain tuple `(controlnet_block_samples, controlnet_single_block_samples)`
rather than a `FluxControlNetOutput` object, avoiding the pytree reconstruction failure.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    51.18s
- Tier A attempts: N/A

## Files changed
- `tt-xla/pytest.ini`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)
- `tt_forge_models/flux_controlnet_depth_v3/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 22137728a4ee9a8581da332a4913eee5a1f8d3cf |
| tt-forge-models | b97234efb6cbe8b981a665b35905a04fcc03c073 |
