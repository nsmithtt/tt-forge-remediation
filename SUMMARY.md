# Remediation Summary: huihui-qwen3-5-9b-abliterated-grimoire-kto-i1-gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_9b_abliterated_grimoire_kto_i1_gguf/causal_lm/pytorch-9B_Abliterated_Grimoire_KTO_i1_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Qwen3.5 hybrid SSM+attention architecture (full_attention_interval=4) requires a new transformers model class; standard Qwen3ForCausalLM cannot represent per-layer variable head counts

## Stack layer
loader

## Tier
B

## Bug fingerprint
qwen35-hybrid-full-attn-interval-size-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

After fixing the loader's `load_gguf_checkpoint` patch signature (see Fix below), the test
surfaces its true blocker:
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error.
  model.layers.3.self_attn.q_proj.weight | MISMATCH |
  ckpt: torch.Size([8192, 4096]) vs model: torch.Size([2048, 4096])
  model.layers.3.self_attn.k_proj.weight | MISMATCH |
  ckpt: torch.Size([1024, 4096]) vs model: torch.Size([512, 4096])
  ... same mismatch at layers 7, 11, 15, 19, 23, 27, 31 ...
```

## Root cause

**Original failure (loader, fixed):** The loader's `_patched_load_gguf_checkpoint` was bound at
module import time to whatever `_gguf_utils.load_gguf_checkpoint` pointed to at that moment —
which was already a broken patcher installed by another loader (using the old two-argument
signature `(gguf_path, return_tensors=False)`). When transformers 5.2.0 began passing
`model_to_load=<model>` as a keyword argument to `load_gguf_checkpoint`, the broken patcher in
the call chain raised `TypeError: unexpected keyword argument 'model_to_load'`. Because all
loaders are imported during pytest collection (via `TorchDynamicLoader.setup_test_discovery()`),
each module-level patch races against others, and whichever ran last held the global reference.

**Terminal failure (loader, Tier B):** The GGUF file declares `general.architecture = "qwen35"` with
`qwen35.full_attention_interval = [4]`. This means every 4th layer (indices 3, 7, 11, …, 31)
uses full multi-head attention with 64 heads (`q_proj` → 8192 × 4096, `k_proj` → 1024 × 4096)
while the remaining 28 layers use GLA (16 heads, `q_proj` → 2048 × 4096). When the architecture
is mapped to `qwen3` for loading into `Qwen3ForCausalLM`, that class assumes uniform head counts
across all layers and initialises every layer with GLA dimensions (16 heads). The full-attention
layers in the checkpoint therefore have weights that cannot be loaded into the model, producing
the size-mismatch RuntimeError. Transformers 5.x has no model class for this hybrid
per-layer-variable-head-count architecture.

## Fix

**Loader fix committed** (`remediation/huihui-qwen3-5-9b-abliterated-grimoire-kto-i1-gguf` branch
in tt_forge_models):

- Added `_unwrap_to_real_load_gguf_checkpoint(fn)`: BFS traversal through patcher chains via
  both `__globals__` (keys `_orig_load_gguf_checkpoint`, `orig_load`) and `__closure__` cells,
  identifying the real transformers function by
  `__module__ == "transformers.modeling_gguf_pytorch_utils"` and
  `__qualname__ == "load_gguf_checkpoint"`.
- Changed `_patched_load_gguf_checkpoint` to use `(*args, **kwargs)` and delegate to
  `_real_load_gguf_checkpoint`, which is the unwrapped real function.
- Added `_apply_gguf_patches()` called at the start of `load_model()` to re-install the correct
  patcher over any broken global patcher that a concurrently imported loader may have installed.
- File: `tt-xla/third_party/tt_forge_models/huihui_qwen3_5_9b_abliterated_grimoire_kto_i1_gguf/causal_lm/pytorch/loader.py`

**Terminal Tier B bug — no fix attempted:**

To load the hybrid Qwen3.5 architecture correctly, transformers would need a new model class
(e.g., `Qwen35HybridForCausalLM`) that initialises `self_attn` projections with
`full_attention_head_count` at every `full_attention_interval`-th layer and GLA projections
elsewhere. This is new infrastructure touching model class registration, config parsing, and
GGUF tensor mapping — well beyond the loader layer.

## Tier B justification

**new-infrastructure** — the fix requires implementing a new model class for the Qwen3.5
hybrid SSM+attention architecture. No existing transformers class can represent per-layer
variable head counts. Changes would span `modeling_qwen3_5.py`, `configuration_qwen3_5.py`,
GGUF architecture registration, and tensor mapping — new code in transformers, not a scoped
loader-layer patch.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/huihui_qwen3_5_9b_abliterated_grimoire_kto_i1_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0cf8ddb40acb4f79232c233f4ba25518f03c6c77 |
| tt-forge-models | f887f92e6383e77209f2e6a5b6b753b959f54da3 |
