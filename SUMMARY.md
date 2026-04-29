# Remediation Summary: eloirotava_restobot_gguf-causal_lm-pytorch-restobot_mario-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[eloirotava_restobot_gguf/causal_lm/pytorch-restobot_mario-single_device-inference]

## Result
FAIL — qwen35 GGUF architecture has no proper transformers tensor mapping for GLA/SSM layers; full-attention layers cause size mismatch when loaded as qwen3; Tier B new-infrastructure in loader

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-gguf-gla-ssm-tensor-mapping-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before our fix):
```
RuntimeError: TT_THROW @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/jit_build/build.cpp:60: tt::exception
info:
brisc build failed. Log: .../riscv-tt-elf/bin/ld: cannot open linker script file
.../third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/runtime/hw/toolchain/blackhole/firmware_brisc.ld: No such file or directory
```

After our tt-xla fix:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.q_proj.weight | MISMATCH | ckpt: torch.Size([4096, 2048]) vs model:torch.Size([1024, 2048])
model.layers.{3, 7, 11, 15, 19, 23}.self_attn.k_proj.weight | MISMATCH | ckpt: torch.Size([512, 2048]) vs model:torch.Size([256, 2048])
(etc.)
```

## Root cause

**Bug 1 (fixed): tt-xla infrastructure — `firmware_brisc.ld` missing**

`setup_tt_metal_home()` in `pjrt_plugin_tt/__init__.py` set `TT_METAL_RUNTIME_ROOT` to the tt-metal source tree path (via `TTMLIR_TTMETAL_SOURCE_DIR` symlink). The directory existed but lacked `runtime/hw/toolchain/` since the hw toolchain build outputs are written in-source and then installed to `third_party/tt-mlir/install/tt-metal/`. The JIT BRISC firmware linker scripts live in the install directory, not the source tree, causing the `firmware_brisc.ld: No such file or directory` error at device kernel compilation time.

**Bug 2 (unfixed, Tier B): loader — qwen35 GGUF GLA/SSM tensor mapping missing**

The `restobot_mario.gguf` file uses GGUF architecture `qwen35`, which is a hybrid GatedDeltaNet (GLA) + full self-attention model with `full_attention_interval=4`. Transformers 5.2.0 does not include `qwen35` in `GGUF_CONFIG_MAPPING`. The `tvall43_qwen3_5_*` loaders register a global patch (`_patch_qwen35_support`) that maps `qwen35` → `qwen3`, but this is wrong: the full-attention layers at indices {3,7,11,15,19,23} have 32 q_heads / 4 kv_heads in the GGUF, while a `Qwen3Config` instantiated from the global `head_count=8` metadata has only 8 q_heads / 2 kv_heads. This causes `RuntimeError` in `_finalize_model_loading` due to shape mismatches on those layers.

The correct fix requires:
1. Adding `qwen35` → `Qwen3_5TextConfig` to `GGUF_CONFIG_MAPPING` in transformers (with `full_attention_interval` correctly read from GGUF metadata)
2. Adding GGUF tensor name mappings in `get_gguf_hf_weights_map` for `Qwen3_5ForCausalLM` parameters (`linear_attn.A_log`, `linear_attn.in_proj_qkv`, `linear_attn.conv1d`, `linear_attn.out_proj`, `linear_attn.norm`, `linear_attn.in_proj_b`, `linear_attn.in_proj_a`, `linear_attn.dt_bias`) to the corresponding `QWEN35` arch tensor types in gguf-py (`SSM_A`, `ATTN_QKV`, `SSM_CONV1D`, `SSM_OUT`, `SSM_NORM`, `SSM_BETA`, `SSM_ALPHA`, `SSM_DT`)
3. The `eloirotava_restobot_gguf` loader should include its own `qwen35` patch rather than relying on side effects from `tvall43_qwen3_5` loaders

Note: gguf-py already has `MODEL_ARCH.QWEN35` with full tensor name mappings for both SSM and attention tensors.

## Fix

**Bug 1 fix (committed):** `tt-xla/python_package/pjrt_plugin_tt/__init__.py`, `setup_tt_metal_home()` — added guard `(tt_metal_path_in_source / "runtime" / "hw" / "toolchain").exists()` on the source-tree path check, and added a new fallback to `third_party/tt-mlir/install/tt-metal/` before raising `FileNotFoundError`. This ensures the install directory (which always has the built hw toolchain artifacts) is found when `TTMLIR_TTMETAL_SOURCE_DIR` is used and the source tree lacks the built toolchain outputs.

Commit: `f6e55d734f3231ca42a286c3b9ed8ea7c09ed725` on branch `remediation/eloirotava_restobot_gguf-causal_lm-pytorch-restobot_mario-single_device-inference` in tt-xla.

**Bug 2 (proposed fix):** In the `eloirotava_restobot_gguf` loader, add a `_patch_qwen35_support()` that:
- Registers `qwen35` in `GGUF_CONFIG_MAPPING` → `Qwen3_5TextConfig` (with `model_type="qwen3_5_text"`)
- Adds `MODEL_ARCH.QWEN35` tensor name mappings for `Qwen3_5ForCausalLM` parameters (bridging gguf-py's `QWEN35` arch names to HF parameter paths)
- Handles `full_attention_interval` from GGUF metadata to build correct `layer_types`

Alternatively, upstream transformers should be patched to add native `qwen35` GGUF support.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure — The fix requires adding new GGUF tensor name mappings for `Qwen3_5ForCausalLM`'s GLA/SSM layers to transformers' GGUF loading infrastructure. gguf-py has the QWEN35 tensor name map but transformers has no bridge from `Qwen3_5ForCausalLM` parameter names (`linear_attn.A_log`, `linear_attn.in_proj_qkv`, etc.) to these GGUF tensor types. This spans the transformers package (GGUF_CONFIG_MAPPING, get_gguf_hf_weights_map) and potentially the loader.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    298.98s (0:04:58)
- Tier A attempts: 1 (pjrt_plugin_tt fix, addressed first bug; second bug is Tier B)

## Files changed
- `tt-xla/python_package/pjrt_plugin_tt/__init__.py` — `setup_tt_metal_home()`: added `runtime/hw/toolchain` subdirectory check and install-dir fallback

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f6e55d734f3231ca42a286c3b9ed8ea7c09ed725 |
| tt-forge-models | 8ee4478d81347d8744d848ee5638b3a97a5c7f82 |
