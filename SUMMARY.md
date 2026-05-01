# Remediation Summary: emotion_recognition_wav2vec2-pytorch-IEMOCAP-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[emotion_recognition_wav2vec2/pytorch-IEMOCAP-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: `length_to_mask` calls `.item()` on a TT tensor to compute `max_len`

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Full failure chain (with speechbrain missing from venv):
```
ModuleNotFoundError: No module named 'speechbrain'
```

After installing `speechbrain` (which pulled `torchaudio 2.11.0` requiring CUDA 13):
```
OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so
libcudart.so.13: cannot open shared object file: No such file or directory
```

After installing `torchaudio==2.9.1+cpu` matching the environment's `torch==2.9.1+cpu`:
```
ValueError: Error code: 13
```

The `swigvarlink` DeprecationWarning is printed as a footnote by pytest after the `1 failed` summary line; it is a side-effect of importing SWIG-based extensions in torchaudio, not the primary error.

## Root cause

**Layer 1 — loader**: The `emotion_recognition_wav2vec2/pytorch/` loader has no `requirements.txt`. When the test is run in a fresh environment, `speechbrain` is absent and the loader fails immediately with `ModuleNotFoundError`. When `pip install speechbrain` is run without the project's extra-index-url, it pulls `torchaudio 2.11.0` (latest on PyPI), which requires CUDA 13 and is incompatible with the `torch==2.9.1+cpu` build in the venv.

**Layer 2 — tt-xla (Tier B)**: After installing the correct `torchaudio==2.9.1+cpu`, the model loads and compilation proceeds. Inside the wav2vec2 forward pass, `speechbrain.integrations.huggingface.huggingface:make_padding_masks` calls `speechbrain.dataio.dataio:length_to_mask`, which contains:

```python
# speechbrain/dataio/dataio.py:836
max_len = length.max().long().item()  # device-to-host transfer
```

The `.item()` call on a TT tensor triggers a device-to-host data transfer that the TT PJRT backend does not support, producing `ValueError: Error code: 13` in `torch_xla._XLAC._xla_warm_up_cache` during `partition_fx_graph_for_cpu_fallback`.

## Fix

**Loader fix (committed)**: Added `requirements.txt` to `emotion_recognition_wav2vec2/pytorch/` listing `speechbrain>=1.1.0` so the dependency is declared. Committed on branch `remediation/emotion_recognition_wav2vec2-pytorch-IEMOCAP-single_device-inference` in tt-forge-models at commit `d4ee6427f07f8035e04eaaa468af01e4489b06e3`.

**Remaining bug (unfixed, Tier B)**: The `.item()` call in `speechbrain/dataio/dataio.py:836` inside `length_to_mask` requires the TT PJRT backend to support device-to-host tensor transfer. The proposed fix would be to add `.item()` / scalar-extraction support to the TT PJRT device implementation, which is new infrastructure.

## Tier B justification

Indicator: **new-infrastructure** — supporting `.item()` (scalar extraction from TT device tensors) requires implementing the device-to-host transfer path in the TT PJRT backend. This is not a scoped pattern guard or single missing lowering — it is a new runtime capability. Same root cause as the known `pjrt-device-to-host-transfer` Tier B bug.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    52.78s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/emotion_recognition_wav2vec2/pytorch/requirements.txt` (added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d4ee6427f07f8035e04eaaa468af01e4489b06e3 |
