# Remediation Summary: a0l_12b_heretic_i1_gguf-causal_lm-pytorch-12B_heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[a0l_12b_heretic_i1_gguf/causal_lm/pytorch-12B_heretic_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
missing-gguf-requirements-txt

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
The `a0l_12b_heretic_i1_gguf/causal_lm/pytorch/` loader directory had no
`requirements.txt`, so it never declared `gguf>=0.10.0` as a dependency.

In a full pytest session the `RequirementsManager` installs `gguf` before
running a model that lists it and then uninstalls it on `__exit__`. When
the a0l test runs after such a test, gguf is absent and
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raises the
`ImportError`. The model itself (LLaMA-architecture, 12B Q4_K_M GGUF from
`mradermacher/A0l-12B-heretic-GGUF`) loads and infers correctly once gguf
is present.

## Fix
Added `a0l_12b_heretic_i1_gguf/causal_lm/pytorch/requirements.txt` with:
```
gguf>=0.10.0
```

- `tt_forge_models` commit `1f505261bf` on branch
  `remediation/a0l_12b_heretic_i1_gguf-causal_lm-pytorch-12B_heretic_GGUF-single_device-inference`
- `tt-xla` commit `06ffdcdc50` on branch
  `remediation/a0l_12b_heretic_i1_gguf-causal_lm-pytorch-12B_heretic_GGUF-single_device-inference`
  (advances `third_party/tt_forge_models` pointer to `1f505261bf`)

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    835.91s (0:13:55)
- Tier A attempts: N/A

## Files changed
- `a0l_12b_heretic_i1_gguf/causal_lm/pytorch/requirements.txt` (new file)

## Submodule hashes
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | 06ffdcdc50d00ba39ad8697cca7c0806b5af8bb6   |
| tt-forge-models | 1f505261bff7693561268e3283e895cd6678c491   |
