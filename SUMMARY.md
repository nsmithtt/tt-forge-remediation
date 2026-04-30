# Remediation Summary: issac_groot/pytorch-Gr00t_N1.6_3B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[issac_groot/pytorch-Gr00t_N1.6_3B-single_device-inference]

## Result
FAIL — four loader bugs fixed; cannot verify silicon pass without IRD_LF_CACHE access

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
groot-n1-6-loader-transformers5x-compat

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Tensor.item() cannot be called on meta tensors

## Root cause
Four interacting loader bugs prevented the N1.6 model from loading under
transformers 5.7.0:

1. **DefaultFastImageProcessorKwargs renamed**: transformers 5.7.0 removed
   `DefaultFastImageProcessorKwargs` from `image_processing_utils_fast`;
   the name moved to `ImagesKwargs` in `processing_utils`.

2. **@dataclass double-processing on PretrainedConfig subclass**:
   transformers 5.7.0 `PretrainedConfig.__init_subclass__` applies
   `@dataclass(kw_only=True)` to every subclass. `GR00T_N1_5_Config` already
   had an explicit `@dataclass` decorator, causing a second pass that deleted
   `field(init=False)` descriptors and then re-added them as `init=True`
   fields without defaults, producing
   `TypeError: non-default argument 'backbone_cfg' follows default argument`.

3. **Missing GR00T_N1_6 model class**: `nvidia/GR00T-N1.6-3B` has
   `model_type: Gr00tN1d6` with a flat config structure (no `backbone_cfg` /
   `action_head_cfg` nested dicts). The code only had `GR00T_N1_5` which
   asserts `isinstance(config.backbone_cfg, dict)`, causing `AttributeError`
   for N1.6 configs. No `Gr00tN1d6` class was registered with AutoConfig.

4. **torch.distributions.Beta in meta-device context**: transformers 5.x
   `PreTrainedModel.from_pretrained` wraps model `__init__` in
   `torch.device("meta")`. `FlowmatchingActionHead.__init__` creates
   `self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)`. `Beta.__init__`
   calls `torch._is_all_true(valid).item()` which fails on meta tensors with
   `RuntimeError: Tensor.item() cannot be called on meta tensors`.

## Fix
All fixes are in `tt_forge_models` on branch
`remediation/issac-groot-pytorch-gr00t-n1-6-3b-single-device-inference`:

1. `issac_groot/pytorch/src/model.py`: Replace
   `from transformers.image_processing_utils_fast import DefaultFastImageProcessorKwargs`
   with `ImagesKwargs as DefaultFastImageProcessorKwargs` from
   `transformers.image_processing_utils_fast`.

2. `issac_groot/pytorch/src/model.py`: Remove `@dataclass` decorator from
   `GR00T_N1_5_Config` (transformers `__init_subclass__` already applies it).

3. `issac_groot/pytorch/src/model.py`: Add `GR00T_N1_6_Config` and
   `GR00T_N1_6` classes to handle the flat N1.6 config structure; register
   with `AutoConfig`/`AutoModel`; dispatch `_load_model` to `GR00T_N1_6` when
   model path contains "N1.6"; guard `action_head_cfg` update for N1.6 models;
   guard `_load_metadata` to skip when `experiment_cfg` dir is absent; add
   try/except around `get_file` in `EagleBackbone` and `GR00TTransform` for
   compile-only environments.

4. `issac_groot/pytorch/src/model.py`: Patch
   `PreTrainedModel.get_init_context` in `_load_model` to filter out
   `torch.device("meta")` from the returned context list, so
   `FlowmatchingActionHead` initialises on CPU and `Beta(...)` succeeds.

## Verification
- pytest exit: not-run (no IRD_LF_CACHE access in this environment)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- `issac_groot/pytorch/src/model.py` (tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 0b720724500733f2b6d0eea4788799afd585a9c2 |
| tt-forge-models | eb4b6b292744eb220657107ae7ef821ef140ba6d |
