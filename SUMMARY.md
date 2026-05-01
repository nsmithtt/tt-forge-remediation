# Remediation Summary: magi-object_detection-pytorch-magi-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[magi/object_detection/pytorch-magi-single_device-inference]

## Result
SILICON_PASS — 4 loader bugs + 1 tt-mlir SharedLHSMatmulFusion bias-rank bug fixed

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
ttir-shared-lhs-matmul-fusion-cse-bias-rank-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
The image processor of type `ViTImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor.
```
Then (after loader fixes): `loc("add.7213"): error: 'ttir.concat' op All input tensors must have the same rank.`

## Root cause
**Loader (4 bugs in tt_forge_models + 1 in tt-xla):**

1. `MagiConfig.__init__` passes `detection_model_config` as a plain dict to the parent class.
   In transformers 5.x, `PretrainedConfig.from_dict()` always returns a base `PretrainedConfig`
   regardless of `model_type`, so the backbone config lost its ConditionalDetrConfig subclass.
   Fix: explicitly call `ConditionalDetrConfig(**detection_model_config)` in a patched
   `MagiConfig.__init__`.

2. `MagiModel.__init__` does not call `self.post_init()`. transformers 5.x requires `post_init()`
   to set `all_tied_weights_keys`. Fix: patch `MagiModel.__init__` to call `self.post_init()`.

3. `MagiModel` has no `forward()` method, so `PreTrainedModel._forward_unimplemented` raises
   `NotImplementedError`. Fix: patch `MagiModel.forward` to route to
   `_get_detection_transformer_output`.

4. transformers 5.x renamed backbone parameter paths from `backbone.conv_encoder.model.*` to
   `backbone.model.*`. The checkpoint uses the old format, so without remapping the backbone
   weights were randomly initialized. Fix: post-load key remapping from safetensors checkpoint.

5. `dynamic_loader.py` inserted `models_root` into `sys.path`, causing `tt_forge_models/spacy/`
   to shadow the real spaCy package and corrupt `sys.modules['spacy']`, making `datasets._dill`
   fail to load. Fix: remove the `sys.path.insert`.

**Compiler (tt-mlir Tier A bug):**

`MatmulWithBiasFusionPattern` converts `matmul + add(bias)` to `LinearOp`. In the Magi decoder,
6 layers share the same `query_position_embeddings.weight` as LHS. After
`MatmulWithBiasFusionPattern`, each LinearOp's bias is a reshape/broadcast chain. The subsequent
`createCanonicalizerPass` CSE-deduplicates the broadcast across 6 uses into a single SSA value
`%1562: tensor<1x305x256xbf16>` (rank 3). `peelBiasTransformations` cannot peel multi-use values,
so one LinearOp's bias remains rank-3 while the others peel to rank-1.

`SharedLHSMatmulFusion<LinearOp>::collectCandidates` had no guard for bias rank mismatch, so it
included the rank-3 LinearOp in the same fusion group as the rank-1 ones.
`createConcatenatedBias` then tried to create a `ttir.concat` of rank-1 and rank-3 tensors,
failing the MLIR verifier.

## Fix
**tt_forge_models** (`magi/object_detection/pytorch/loader.py`, branch
`remediation/magi-object_detection-pytorch-magi-single_device-inference`):
- Commit `19f9c98435`: patch MagiConfig.__init__, MagiModel.__init__, MagiModel.forward (fixes 1–3)
- Commit `0171f2a69f`: post-load remap of backbone keys from safetensors (fix 4)

**tt-xla** (`tests/runner/utils/dynamic_loader.py`, branch
`remediation/magi-object_detection-pytorch-magi-single_device-inference`):
- Commit `5f9760c53`: remove sys.path.insert(models_root) that shadows spacy package (fix 5)

**tt-mlir** (`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`, branch
`remediation/magi-object_detection-pytorch-magi-single_device-inference`):
- Commit `0c30c41c1`: guard SharedLHSMatmulFusion on output rank mismatch (pre-existing guard)
- Commit `3a9809dc3`: guard SharedLHSMatmulFusion against CSE-deduplicated bias rank mismatch

In `collectCandidates`, added a bias rank guard for `LinearOp`: if `rootBiasRank != candidateBiasRank`, the candidate is skipped. This prevents the rank-3 CSE-shared bias LinearOp from being included in a fusion group with rank-1 bias LinearOps.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    135.53s (0:02:15)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/magi/object_detection/pytorch/loader.py`
- `tt-xla/tests/runner/utils/dynamic_loader.py`
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 3a9809dc376bb170d1b56e93b897632b7bcdb808 |
| tt-xla          | 5f9760c534f879df0f14cade19cedcfab40daa90 |
| tt-forge-models | 0171f2a69f4850e632a06062474be253fef66d73 |
