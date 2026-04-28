# Remediation Summary: mlx_community_gemma_3n_e2b_it_4bit-multimodal-pytorch-E2B_IT_MLX_4bit-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[mlx_community_gemma_3n_e2b_it_4bit/multimodal/pytorch-E2B_IT_MLX_4bit-single_device-inference]

## Result
FAIL — `concat-cb-size-exceeds-l1`: `ttnn::prim::ConcatDeviceOperation` allocates 4,305,408 B of CBs per core (2.7× L1 limit); same Tier B bug as egoactor report (commit ac486ec)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
concat-cb-size-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=2)] grow to 4305408 B which is beyond max L1 size of 1572864 B
```
followed by:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two issues were found and partially fixed:

**Issue 1 (fixed — loader):** The `mlx-community/gemma-3n-E2B-it-4bit` safetensors checkpoint
uses MLX-native 4-bit quantization (packed `uint32` with 8 int4 values per element,
LSB-first) and NHWC conv weight layout. The original loader called `from_pretrained` directly,
which triggered a `ValueError` because the MLX `quantization_config` has no `quant_method` key.
Even after deleting `quantization_config`, the packed shapes were incompatible with the
initialized architecture. Fixed by:
- Deleting `config.quantization_config` before model init so the transformer dispatcher
  skips all quantizer paths.
- Implementing `_load_mlx_state_dict`: reads every tensor from `model.safetensors`,
  dequantizes packed-int4 weights using companion `.scales`/`.biases` tensors
  (`original = packed_int4 * scale + bias`), transposes 4-D conv weights from
  NHWC `[out,H,W,in]` → NCHW `[out,in,H,W]` and 1-D conv weights from `[ch,K,1]`
  → `[ch,1,K]`, and remaps key names (`language_model.model.X` → `model.language_model.X`).
- CPU forward pass verified: logits shape `[1, 277, 262400]`.

**Issue 2 (unfixed — tt-metal, Tier B):** After the loader fix the model runs on the TT device
through `partition_fx_graph_for_cpu_fallback`. During probing of a subgraph from
`Gemma3nModel.forward` (the resume segment after a graph break at `torch_compilable_check`
inside `get_placeholder_mask`), `ttnn::prim::ConcatDeviceOperation` allocates
4,305,408 bytes of static circular buffers on a 3-core grid `[(x=0,y=0)-(x=0,y=2)]`,
which exceeds the 1,572,864 byte L1 per core. The TT device enters error state 13 and
all subsequent `torch_xla.sync()` calls fail with
`RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

The concat CB overflow arises because the `ConcatDeviceOperation` program factory
allocates CBs sized for the full per-core shard without bounding them to L1 capacity.
This is the same root cause and bug fingerprint (`concat-cb-size-exceeds-l1`) previously
identified in the egoactor_8b_qwen3vl_i1_gguf report (commit ac486ec in this repo).

## Fix
**Loader fix** committed to
`remediation/mlx_community_gemma_3n_e2b_it_4bit-multimodal-pytorch-E2B_IT_MLX_4bit-single_device-inference`
in tt-forge-models (`8f20989b38`), pointed to from tt-xla commit `bae83a903`:
- `mlx_community_gemma_3n_e2b_it_4bit/multimodal/pytorch/loader.py`:
  `_mlx_dequantize()` — dequantizes packed int4 weights using scales/biases.
  `_remap_mlx_key()` — fixes `language_model.model.X` → `language_model.X` + prepends `model.`.
  `_load_mlx_state_dict()` — orchestrates dequantization, NHWC transpose, and key remapping.
  `load_model()` — uses `_load_mlx_state_dict` instead of `from_pretrained`.

**Proposed fix for Issue 2** (not attempted): In `tt-metal`,
`ttnn/cpp/ttnn/operations/data_movement/concat/device/` program factories need to bound
per-core CB allocations to L1 capacity. For large tensors the factories should either
distribute work across more cores or use a streaming approach that limits the per-core
CB footprint.

## Tier B justification
Tier B indicator: `cross-cutting` — fixing the concat CB overflow requires restructuring
CB allocation logic across multiple program factory files in tt-metal
(`concat_program_factory.cpp`, `concat_s2s_rm_program_factory.cpp`,
`concat_s2i_program_factory.cpp`, etc.) to cap per-core allocations within the 1.5 MB L1
budget, which is complex multi-core kernel scheduling logic affecting all concat paths.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    486.41s (0:08:06)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: mlx_community_gemma_3n_e2b_it_4bit/multimodal/pytorch/loader.py` (complete rewrite of load path)
- `tt-xla: third_party/tt_forge_models` (submodule pointer updated to `8f20989b38`)

## Submodule hashes
| Submodule       | Commit                                   |
|-----------------|------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | bae83a9039359d1f0a8632a0c406842e57c1f993 |
| tt-forge-models | 8f20989b38c1da8256b305ed7bd0bfaee703ce09 |
