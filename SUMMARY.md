# Remediation Summary: huihui_lfm2_24b_a2b_abliterated_i1_gguf-causal_lm-pytorch-24B_A2B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch-24B_A2B_GGUF-single_device-inference]

## Result
XFAIL — LFM2 24B MoE dequantizes to ~48 GB BF16, exceeding single-device DRAM on all supported hardware

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-24b-moe-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   NotImplementedError: "histogram_cpu" not implemented for 'Int'
```
(In CI sessions where another LFM2 loader has already registered the `lfm2moe` GGUF
architecture. In an isolated run the first error is:
`ValueError: GGUF model with architecture lfm2moe is not supported yet.`)

## Root cause

**Bug 1 — loader (fixed):** `lfm2moe` GGUF architecture not registered in transformers
5.2.x. Only `lfm2` (dense variant) appears in `GGUF_SUPPORTED_ARCHITECTURES` and
`GGUF_TO_TRANSFORMERS_MAPPING`. The 24B MoE GGUF files declare
`general.architecture = "lfm2moe"`.

**Bug 2 — loader (fixed):** `grouped_mm_experts_forward` in
`transformers/integrations/moe.py` branches on `device.type != "cpu"` to choose
`expert_ids_g.int()` for `torch.histc`. Under TT's XLA device (`device.type = "xla"`)
the int path is chosen, but `torch.histc` CPU fallback rejects integer input:
`NotImplementedError: "histogram_cpu" not implemented for 'Int'`.

**Bug 3 — hardware capacity:** LFM2 24B A2B is a 24-billion-parameter MoE model (40
layers, 64 experts). The GGUF Q4_K_M file is ~13 GB but transformers dequantizes all
weights to BF16 at load time: 24B × 2 bytes ≈ 48 GB. This far exceeds the DRAM of all
supported single-device hardware (n150: 12 GB, p150b: 24 GB). The prior bartowski
LFM2-24B-A2B-GGUF remediation (same underlying model) confirmed the model progresses
past loader bugs but subsequently hits a Tier B segfault during
`partition_fx_graph_for_cpu_fallback` — still before silicon execution — which means the
hardware capacity ceiling was never directly measured. However, 48 GB > 24 GB (p150b) is
deterministic; the test is correctly XFAIL.

## Fix

**Loader fixes applied in tt_forge_models:**

1. **`requirements.txt`** — added `gguf>=0.10.0`.

2. **`_register_lfm2moe_gguf_arch()`** — registers `lfm2moe` in
   `GGUF_SUPPORTED_ARCHITECTURES`, `TENSOR_PROCESSORS`,
   `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, and `GGUF_TO_FAST_CONVERTERS`. Called at
   module import time so `AutoConfig.from_pretrained` works during test collection.

3. **`_patch_grouped_mm_experts_forward()`** — replaces `grouped_mm_experts_forward`
   with a version that calls `expert_ids_g.float()` when `device.type != "cuda"`. Both
   `moe_module.grouped_mm_experts_forward` and
   `moe_module.ExpertsInterface._global_mapping["grouped_mm"]` are updated so the
   `ExpertsInterface` dispatch picks up the patch.

4. **`_apply_lfm2moe_load_patches()`** — wraps `load_gguf_checkpoint` at all import
   sites to remap `model_type: lfm2moe → lfm2_moe` and build `layer_types` from the
   per-layer `num_key_value_heads` list (MoE uses `layer_types`, not `full_attn_idxs`).
   Uses `_find_base_load_gguf()` BFS over `sys.modules` to bypass broken intermediate
   wrappers from other loaders that omit `**kwargs`.

5. **Token embedding resize** — after `from_pretrained`, resize embeddings when the
   tokenizer's vocab is larger than `model.config.vocab_size` (GGUF tokenizer can add
   special tokens not counted in the config's `vocab_size`).

**Test config update applied in tt-xla:**

Both variants (`24B_A2B_GGUF` and `24B_A2B_i1_GGUF`) marked `KNOWN_FAILURE_XFAIL` in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: not-run (XFAIL marked before hardware run)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-forge-models`: `huihui_lfm2_24b_a2b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (updated)
- `tt-xla`: `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entries added)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a51f0fc0c25051b17229f1c34002b40c0afb85b0 |
| tt-forge-models | caeec6424c80e3966f5a6ac1fa5d5be6deee23ba |
