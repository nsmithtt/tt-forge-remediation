# Remediation Summary: pig_encoder-pytorch-clip_l_f16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[pig_encoder/pytorch-clip_l_f16-single_device-inference]

## Result
NO_FIX_NEEDED — failure could not be reproduced; test passes on configured branch

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-21 22:46:46.310 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
The "Fabric Router Sync: Timeout" is a transient hardware initialization failure in the tt-metal fabric router. It occurs when the device (or a neighboring device in the chassis) is left in a bad state from a prior test crash — the fabric router's sync handshake times out before reaching the expected `0xa2b2c2d2` state.

Inspection of results_main.yaml shows that on the same host (`120-qb2-p04t05-tt-xla-dev`) around the same time, the prior test `taobao_mnn_qwen3_5_9b` failed with a `timeout` at 22:25, which likely left the device in a bad state. The pig_encoder test ran at 22:47 and hit the fabric router timeout on device initialization. Subsequent tests on the same host (flux_krea_dev_comfyui at 23:31, langcache_embed_v3_small at 23:59) passed normally after the CI framework reset the device.

The pig_encoder model itself is a standard CLIPTextModel (openai/clip-vit-large-patch14), approximately 123M parameters. It loads and runs without issue on a clean device, as confirmed by reproduction on this host: SILICON_PASS in 66.78s.

## Fix
No fix required. This is a transient infrastructure issue, not a model or compiler bug.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    66.78s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 7fac71a226e6935869162589289ff6ec8cd9b090 |
