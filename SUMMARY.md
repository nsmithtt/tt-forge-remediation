# Remediation Summary: florence_2-image_captioning-pytorch-Base-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[florence_2/image_captioning/pytorch-Base-single_device-inference]

## Result
FAIL — TTNN SDPA kernel does not support k_len=1 (decoder self-attention with one token violates k_chunk_size >= 32 constraint)

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Verbose logs show:
```
ERR| Exception:
 --- ttnn::prim::SDPAOperation (enqueue_mesh_workload)
```

## Root cause
**Layer**: tt-metal / tt-mlir (TTNN SDPA kernel)

Florence-2 is an encoder-decoder model. The test configuration feeds a single decoder token (`decoder_input_ids = [[eos_id]]`), so the decoder self-attention has q_len=1 and k_len=1:

```
q=[1, 16, 1, 64], k=[1, 16, 1, 64]   # decoder self-attention, step 1
q=[1, 16, 1, 64], k=[1, 16, 579, 64]  # decoder cross-attention (fine, k=579)
```

The TTNN SDPA kernel requires `k_chunk_size >= 32`. With k_len=1 the kernel throws an internal error (Error code: 13).

The loader failure (`AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`) was a pre-existing transformers 5.x incompatibility fixed in commit `bb4d183ea2` on `origin/arch-c-36-tt-xla-dev/nsmith/hf-bringup-0`. That fix uses the native `Florence2ForConditionalGeneration` class (no `trust_remote_code`) with synthetic inputs to bypass the HuggingFace remote-code incompatibility.

After applying `bb4d183ea2`, the loader issue is resolved and the SDPA constraint violation is exposed.

## Fix
The bug lives in the **tt-metal / tt-mlir layer** (TTNN SDPA kernel). A real fix would need to:

- Support k_len < 32 in the TTNN SDPA kernel (e.g., by padding keys to the minimum chunk size of 32 and masking the padding), OR
- Fall through to a non-chunked attention path for short sequences.

This cannot be fixed in the loader: increasing `decoder_input_ids` sequence length to pad to 32 would change the model semantics and is listed as a forbidden workaround. The decoder starts from one token and the k_len=1 self-attention is the correct first step.

## Verification
Not reached — test fails at SDPA execution before any PCC comparison.

Wall-clock time to failure: ~105 s
Hardware: n150 (Wormhole B0)

## Files changed
- `third_party/tt_forge_models/florence_2/image_captioning/pytorch/loader.py`
  (via commit `bb4d183ea2` on `origin/arch-c-36-tt-xla-dev/nsmith/hf-bringup-0` — loader ported to native transformers Florence2ForConditionalGeneration)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f88a55ccd4274dd36dbca44b3d32343910f0ec41 |
| tt-forge-models | 9260e41d8bee311f5085106958e9a247bd9832de |
