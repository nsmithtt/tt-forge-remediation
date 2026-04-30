# Remediation Summary: deepseek-deepseek_v3_5layer-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_5layer/pytorch-single_device-inference]

## Result
FAIL — static MoE unrolling (256 experts × 5 layers) places all expert weights in device DRAM (34.2 GB limit exceeded) because dynamic routing is untraceable on TT

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
moe-static-unroll-dram-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original (before loader fix): `ModuleNotFoundError: No module named 'triton'`

After loader fix (on silicon): `TT_FATAL: Out of Memory: Not enough space to allocate 15091105792 B DRAM buffer across 8 banks, where each bank needs to store 1886388224 B, but bank size is 4273390016 B (allocated: 2651351936 B, free: 1622038080 B, largest free block: 1602578752 B)` → `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

## Root cause

Four loader bugs were present and fixed:

1. **`ModuleNotFoundError: No module named 'triton'`** — `chwan/DeepSeek-V3-5layer` uses FP8 quantization (`quant_method: fp8`). transformers' `finegrained_fp8.py` imports `triton` unconditionally at module level, crashing before the loader can run. Fix: delete `config.quantization_config` to skip the FP8 quantizer, then manually reload the 128×128 block-wise `weight_scale_inv` tensors and apply them to the BF16 cast weights.

2. **`is_torch_fx_available` removed in transformers 5.x** — the remote `modeling_deepseek.py` imports `is_torch_fx_available` from `transformers.utils.import_utils`, which no longer exists. Fix: inject a shim before the import.

3. **`DynamicCache.get_usable_length` removed in transformers 5.x** — DeepSeek-V3 custom code calls this at inference time. Fix: add the method back.

4. **`moe_infer` uses `.cpu().numpy()` dispatch** — The MoE expert dispatch loop calls `tokens_per_expert.cpu().numpy()` to drive a per-expert for-loop. During FakeTensor tracing this crashes dynamo; on the TT device it triggers PJRT device-to-host transfer (INTERNAL error 13). Fix: replace `moe_infer` with a static per-expert masked matmul over `range(experts_per_rank)`.

After all four loader fixes were applied, the model loads and compiles successfully. However, the static MoE patch unrolls the 256-expert loop into a single graph per layer. With 5 layers, the tt-mlir compiler must handle 5 × 256 × 3 = 3,840 matmul ops (compilation took 42.7 minutes). During device execution, tt-metal's allocator tries to place all expert weight tensors into device DRAM. The device has 8 banks × 4.27 GB = 34.2 GB total DRAM. The expert weights alone are 256 experts × 5 layers × ~83.9 MB = ~105 GB, far exceeding the device limit.

The underlying reason dynamic MoE routing cannot be used is that `moe_infer`'s data-dependent token-count control flow (`.cpu().numpy()` dispatch) is untraceable in TT's static-graph compilation model. The static workaround necessarily puts all expert weights in DRAM.

## Fix
Proposed fix: add dynamic dispatch / conditional routing support in `tt-xla` so only the `num_experts_per_tok` (8) active expert weights need to be in device DRAM per forward pass. With dynamic routing, the device DRAM required is 8 experts × 83.9 MB × 5 layers ≈ 3.4 GB, well within the 34.2 GB limit.

Loader fixes are committed at:
- `tt_forge_models` branch `remediation/deepseek-deepseek_v3_5layer-pytorch-single_device-inference` (commit `a1c2109ef0`)
- `tt-xla` branch `remediation/deepseek-deepseek_v3_5layer-pytorch-single_device-inference` (commit `12756deec8`)

## Tier B justification
The dynamic routing fix requires new infrastructure in `tt-xla`: the PJRT/XLA compilation pipeline currently requires static graphs. Supporting data-dependent expert selection (a conditional/index-based dispatch pattern) requires either (a) PJRT-level dynamic dispatch (new PJRT path) or (b) on-the-fly weight streaming from host memory. Both are cross-repo changes touching multiple files in `tt-xla`, `tt-mlir`, and `tt-metal`.

Tier B indicator: **new-infrastructure** (dynamic routing / conditional expert dispatch)

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b (bh-lb-12, 8 banks × 4.27 GB = 34.2 GB device DRAM)
- Duration: 3517.70s (0:58:37) — of which ~42.7 min was tt-mlir compilation of the 3,840-op static graph
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_5layer/pytorch/loader.py` in `tt_forge_models`
  (is_torch_fx_available shim, DynamicCache.get_usable_length, FP8 weight dequantization, static moe_infer)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 12756deec84400488c27a10fd2795a923d5a40c3 |
| tt-forge-models | a1c2109ef0e8be08c736f572e8df96314b0055ef |
