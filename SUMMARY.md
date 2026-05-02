# Remediation Summary: issac_groot-pytorch-Gr00t_N1.6_DROID-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[issac_groot/pytorch-Gr00t_N1.6_DROID-single_device-inference]

## Result
FAIL — GR00T N1.6 backbone architecture not implemented in loader; requires new `Gr00tN1d6` model class

## Stack layer
loader

## Tier
B

## Bug fingerprint
groot-n1-6-backbone-class-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'GR00T_N1_5_Config' object has no attribute 'backbone_cfg'
```
Triggered during `GR00T_N1_5.from_pretrained("nvidia/GR00T-N1.6-DROID")` in `Gr00tPolicy._load_model` (issac_groot/pytorch/src/model.py:684).

Original CI failure was `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`, which was an OOM/internal error after model compilation on silicon, but the current codebase fails earlier in the loader.

## Root cause
The loader uses a single `GR00T_N1_5` class for all GR00T variants including N1.6. The N1.5 and N1.6 architectures are structurally different:

1. **Config class mismatch**: `GR00T_N1_5_Config` is `@dataclass`-decorated with a `backbone_cfg: dict = field(init=False)` field. The N1.6 `config.json` (`model_type: "Gr00tN1d6"`) has no `backbone_cfg` key. When `from_pretrained` calls `logger.info(config)`, the dataclass-generated `__repr__` tries to access `backbone_cfg`, raising `AttributeError`.

2. **Weight key mismatch**: N1.5 model safetensors use `backbone.eagle_model.*` weight keys, matching `EagleBackbone.eagle_model` attribute. N1.6 uses `backbone.model.*` keys — there is no code to handle this mapping.

3. **Backbone config unavailable**: `EagleBackbone.__init__` calls `get_file("test_files/pytorch/Issac_groot/config.json")` which requires the internal `IRD_LF_CACHE` server. The N1.6 backbone model (`nvidia/Eagle-Block2A-2B-v2`) is not publicly accessible on HuggingFace.

4. **No `Gr00tN1d6` class**: The N1.6 architecture (`"architectures": ["Gr00tN1d6"]`) has no corresponding Python class in the loader — implementing it requires new infrastructure.

Two earlier loader bugs were found and fixed (committed to the remediation branch):
- Fix 1: `DefaultFastImageProcessorKwargs` renamed to `ImagesKwargs` in transformers 5.2.x — fixed with try/except fallback.
- Fix 2: `build_eagle_processor()` called `get_file()` for `vocab.json`/`merges.txt` from `IRD_LF_CACHE` — fixed by using `hf_hub_download('Qwen/Qwen2.5-0.5B', ...)` instead.

## Fix
Two loader fixes were committed to `remediation/issac_groot-pytorch-Gr00t_N1.6_DROID-single_device-inference` in tt_forge_models:
- `35181affe2`: `issac_groot/pytorch/src/model.py` — try/except import for `DefaultFastImageProcessorKwargs`/`ImagesKwargs`
- `24ffe2b090`: `issac_groot/pytorch/src/model.py` — `build_eagle_processor()` rewritten to use `hf_hub_download('Qwen/Qwen2.5-0.5B', ...)`

The terminal bug (backbone class mismatch) would require:
- A new `Gr00tN1d6Config` config class handling the N1.6 JSON format
- A new backbone class with `self.model` attribute (not `self.eagle_model`) to match N1.6 weight keys
- Either access to the IRD_LF_CACHE Eagle 2B config or a public alternative
- Registering `Gr00tN1d6` in transformers `AutoConfig`/`AutoModel` if needed

## Tier B justification
`new-infrastructure`: The N1.6 model requires a new `Gr00tN1d6` model class (not a modification of `GR00T_N1_5`), a new backbone class with different attribute naming to match the `backbone.model.*` weight key scheme, and a config that the internal `IRD_LF_CACHE` server must serve since the backbone model (`nvidia/Eagle-Block2A-2B-v2`) is not publicly accessible. This exceeds a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/issac_groot/pytorch/src/model.py` (two loader fixes committed)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 24ffe2b090d7adde5e3b37b5ab3ef265d9c6c64b |
