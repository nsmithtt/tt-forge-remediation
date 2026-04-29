# Remediation Summary: deepseek_r1-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1/pytorch-single_device-inference]

## Result
FAIL — loader bug fixed (grouped_mm histc-on-Int); blocked by Tier B compiler-stack bug ttmlir-3d-gather-large-page-l1-overflow after fix

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-3d-gather-large-page-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
NotImplementedError: "histogram_cpu" not implemented for 'Int'

While executing %histc : [num_users=1] = call_function[target=torch.ops.aten.histc.default](args = (%_to_copy_62, 256, 0, 255), kwargs = {})
Original traceback:
  ...
  File ".../transformers/integrations/moe.py", line 271, in grouped_mm_experts_forward
    num_tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
```

After loader fix — remaining silicon failure:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` in the reported failure message is the last line of the pytest warnings summary; the actual error was the `grouped_mm histc-on-Int` issue.

## Root cause
Two bugs compound:

**Bug 1 (loader, fixed):** `deepseek-ai/DeepSeek-R1-0528` config has `_experts_implementation = None`. `PreTrainedModel.__init__` calls `get_correct_experts_implementation(None)` which defaults to `"grouped_mm"` when `torch.nn.functional.grouped_mm` is available (PyTorch 2.9+). `grouped_mm_experts_forward` (moe.py:270) picks the Int path for `torch.histc` when `device.type != "cpu"` — but under TT's XLA device (`device.type == "xla"`), CPU histc rejects Int dtype → `NotImplementedError: "histogram_cpu" not implemented for 'Int'`.

**Bug 2 (tt-mlir, Tier B):** After switching to `batched_mm`, the experts forward path accesses `gate_up_proj[expert_ids]` where `gate_up_proj` has shape `[256, 4096, 1024]` (`n_routed_experts=256`, `moe_intermediate_size=2048`, `hidden_size=1024`). tt-mlir lowers this `stablehlo.gather` by reshaping the weight to `[256, 4,194,304]` and calling `ttnn.embedding`. The embedding CB allocates one full weight-table row = `4,194,304 × 2 bytes = 8 MB` per core, which is 5× the Wormhole L1 limit of 1,572,864 B. Surfaces as `INTERNAL: Error code: 13` from `_run_cached_graph`.

## Fix
**Loader fix (applied):** In `deepseek/deepseek_r1/pytorch/loader.py`, after `AutoConfig.from_pretrained`, add:
```python
config._experts_implementation = "batched_mm"
```
This forces `PreTrainedModel.__init__` to use the vectorized `batched_mm` path instead of `grouped_mm`, eliminating the `torch.histc` with Int dtype.

**Proposed compiler-stack fix (Tier B, not implemented):** The `stablehlo.gather` for 3D MoE weight tensors should not be lowered to `ttnn.embedding` after a 2D reshape. Fix would require:
1. A guard in `tt-mlir` gather lowering to detect when the CB page would exceed L1 and skip the embedding path
2. A corresponding change to the tt-metal embedding kernel or a new gather kernel that handles large rows via sub-row streaming

This touches ≥3 files across ≥2 repos (tt-mlir gather lowering + tt-metal embedding kernel + host program factory).

## Tier B justification
Indicator: **cross-repo** — fix requires coordinated changes to tt-mlir gather lowering pattern (1–2 files in `tt-mlir`) and tt-metal embedding kernel or new gather kernel (1–2 files in `tt-metal`). The L1 overflow cannot be avoided by a single scoped pattern guard in one file; the embedding CB page allocation formula would need to be changed alongside the kernel that reads it.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    182.57s (0:03:02) — INTERNAL: Error code: 13 after loader fix
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1/pytorch/loader.py` — add `config._experts_implementation = "batched_mm"` in `load_model` (tt-forge-models repo, branch `remediation/deepseek_r1-pytorch-single_device-inference`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ace1de7b0ab3df9a1cd4dc7bb4c6f69bc30650fc |
| tt-forge-models | 2ea2051ac5d1d5862f83f87218fb5d83dada49a4 |
