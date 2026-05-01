# Remediation Summary: min-dalle-pytorch-mega-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[min_dalle/pytorch-mega-single_device-inference]

## Result
SILICON_PASS — three loader bugs fixed: missing requirements.txt, min_dalle namespace shadowing, and pose_tokens not registered as buffer

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
min-dalle-loader-namespace-shadow-and-missing-requirements

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original CI failure was:
```
2026-04-23 23:19:22.289 | critical | Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)
```
This is a known transient hardware initialization error that CI explicitly excludes from the tt_fatal failure category (documented in failure_patterns.yaml). On fresh reproduction, three real loader bugs surfaced.

**Bug 1** (actual root cause):
```
ImportError: cannot import name 'MinDalle' from 'min_dalle'
(/home/.../tt_forge_models/min_dalle/__init__.py)
```
The `min-dalle` pip package was not listed in `requirements.txt`, so it was never installed. Even after installation, the `dynamic_loader.py` `sys.path.insert(0, models_root)` caused `tt_forge_models/min_dalle/` (the test model directory) to shadow the pip package `min_dalle`.

**Bug 2** (after namespace shadow fix):
```
KeyError: "attribute 'pose_tokens' already exists"
```
`DalleBartEncoder.__init__` stores `pose_tokens` as a plain tensor attribute. `register_buffer` rejects registering over an existing attribute; needed `del` before registering.

**Bug 3** (underlying): `pose_tokens` stored as a plain tensor is skipped by `model.to(device)`, causing device mismatch (`xla:0` vs `cpu`) in Dynamo FakeTensor propagation.

## Root cause
Three independent loader bugs, each blocking the previous from surfacing:

1. **Missing requirements.txt**: `min-dalle` was never declared as a dependency, so it was not installed before the test ran.

2. **Namespace shadowing**: `dynamic_loader.py` (in the `arch-c-36` branch, predating the spacy shadowing fix) inserted `models_root` (= `tt_forge_models/`) into `sys.path[0]`. When `loader.py` did `from min_dalle import MinDalle`, Python found `tt_forge_models/min_dalle/__init__.py` instead of the installed pip package.

3. **pose_tokens not a buffer**: `DalleBartEncoder.__init__` creates `self.pose_tokens = torch.stack([token_indices] * 2)` on `device="cpu"`, stored as a plain tensor. When the test framework moves the model to the XLA device, plain tensor attributes are not moved. Dynamo's FakeTensor device propagation raises `RuntimeError` because `embed_positions` (on `xla:0`) and `pose_tokens` (on `cpu`) disagree.

## Fix
**In `tt-xla` (`remediation/min-dalle-pytorch-mega-single-device-inference`):**

`tests/runner/utils/dynamic_loader.py`: Removed `sys.path.insert(0, models_root)` in `setup_models_path()`. The `models_root` (= `tt_forge_models/`) was being inserted at `sys.path[0]`, causing any model directory with the same name as a pip package to shadow that package. The fix was already applied to main in commit 83e6bd59b but was not present on the `arch-c-36` bringup branch.

**In `tt_forge_models` (`remediation/min-dalle-pytorch-mega-single-device-inference`):**

`min_dalle/pytorch/requirements.txt` (new): Added `min-dalle` so the pip package is installed before the test runs.

`min_dalle/pytorch/loader.py`:
- Switched `models_root` from `tempfile.mkdtemp()` to a persistent `~/.cache/min_dalle` directory to prevent 2.1 GB weight re-downloads filling `/tmp` on each run.
- After `init_encoder()`, deleted the plain `pose_tokens` attribute and re-registered it as a non-persistent buffer via `register_buffer("pose_tokens", ..., persistent=False)`, so `model.to(xla_device)` moves it correctly.

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 108.90s (0:01:48)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`
- `tt_forge_models/min_dalle/pytorch/requirements.txt` (new)
- `tt_forge_models/min_dalle/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7e8728162e439b34f85582b9c3c9ab0dc5de7314 |
| tt-forge-models | 54d3481dc1d6918832f93a2fa6632ef9f7bd3c4a |
