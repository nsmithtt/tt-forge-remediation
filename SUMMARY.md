# Summary: affine/causal_lm/pytorch-SD32-single_device-inference

## Test
`tests/runner/test_models.py::test_all_models_torch[affine/causal_lm/pytorch-SD32-single_device-inference]`

## Original Failure
```
2026-04-19 01:59:01.676 | critical | Always | TT_FATAL: Out of Memory: Not enough space to allocate 262144000 B DRAM buffer across 8 banks, where each bank needs to store 32768000 B, but bank size is 4273390016 B (allocated: 4202914112 B, free: 70475904 B, largest free block: 22282240 B) (assert.hpp:104)
```

## Outcome: Cannot Reproduce

The test cannot be reproduced because the HuggingFace model repository is inaccessible. Both Affine model variants in the loader require HuggingFace authentication:

- `hO61qjpwxu/Affine-20260227-sd32-5H4PmD8ZRB8Bqck9KmmCg9weowf6ZKJaxFNs8Y2TR3q6HgkZ` (SD32 variant — the test target)
- `luis1027/affine-5EtPj7mKQ6arxx8KW3GFTWBzTBia1DyM2vDU1rpNPsRHUk1B` (alternate variant)

Both return `401 Client Error: Repository Not Found` when accessed without an authenticated HuggingFace token.

## What Was Tried

1. Configured the project with branch `ip-172-31-17-155-tt-xla-dev/ubuntu/hf-bringup-range-534-966-0` and built successfully.
2. Ran the test — failed at `load_config()` step (before any TT hardware execution) with `RepositoryNotFoundError`.
3. Verified no HuggingFace token is present on the system (`~/.config/huggingface/token`, `~/.huggingface/token`, env vars).
4. Searched for existing fix branches — none found for this test.

## Root Cause of Cannot-Reproduce

The Affine SD32 model (`hO61qjpwxu/Affine-20260227-sd32-...`) is a private HuggingFace repository. Without an authenticated token that has access to this private repo, the model cannot be downloaded and the OOM failure cannot be triggered or fixed.

## Submodule Hashes

- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-mlir: `cf42a9b982edb4ae9774b535a5de18dddfa5013b`
- tt-xla: `5ee9f150dbe312d1d717f6da1aa11ac051700b02`
