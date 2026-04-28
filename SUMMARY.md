# Remediation Summary: chexagent/pytorch-chexagent_2_3b-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[chexagent/pytorch-chexagent_2_3b-single_device-inference]

## Result
FAIL — SDPA kernel hangs on Blackhole silicon during SigLIP encoder execution

## Failure
```
Fatal Python error: Segmentation fault
```
(Original failure; after loader fixes, the test runs through CPU inference successfully
and reaches the silicon execution phase, where it hangs indefinitely at the first SDPA
operation in the SigLIP ViT-L encoder.)

## Root cause

**Layer: tt-metal / tt-mlir (compiler-stack)**

Two categories of failures were present:

### 1. Loader-layer bugs (all fixed)

CheXagent-2-3b uses transformers 4.40-era custom remote code; eleven issues
were found and fixed when running under transformers 5.x:

1. `is_tf_available` removed from `transformers.utils` — added False-returning stub.
2. `pad_token_id` missing from `CheXagentConfig.__init__` — patched config.
3. `rope_scaling` auto-populated with new `rope_type` dict by transformers 5.x — reset to None.
4. `CLIPModel.__init__` calls `AutoModel.from_pretrained` inside the meta-device context
   used by the outer `from_pretrained` — fixed by temporarily removing the DeviceContext
   from the TorchFunctionMode stack.
5. DeviceContext restore in the `finally` block used wrong order (re-pushed below other modes
   instead of on top) — fixed.
6. `apply_chat_template` returns a `BatchEncoding` in transformers 5.x, not a raw tensor —
   fixed by extracting `.input_ids`.
7. `_decode()` positional signature changed in transformers 5.x — fixed by passing
   `spaces_between_special_tokens` as a keyword argument.
8. `sample_image_url` pointed to a Wikipedia URL returning 403 — changed to an accessible
   HuggingFace datasets URL.
9. `SiglipVisionEmbeddings.position_ids` is a `persistent=False` buffer; transformers'
   `_move_missing_keys_from_meta_to_device` unconditionally replaces all non-persistent
   buffers with `torch.empty_like()` (garbage data), causing `IndexError` in the embedding
   lookup — fixed by re-creating `position_ids` in `load_model()` after `from_pretrained`
   returns.
10. `CLIPModel.forward` used `self.model(x, output_hidden_states=True).hidden_states[-1]`;
    in transformers 5.x `SiglipEncoder` no longer collects intermediate hidden states so
    `hidden_states` is always `None` — replaced with direct
    `self.model.encoder(inputs_embeds=self.model.embeddings(x)).last_hidden_state` call.
11. The inner `AutoModel.from_pretrained` for SigLIP loads weights as float32; the outer
    CheXagent `from_pretrained` (called with `dtype=bfloat16`) uses `copy_()` for non-meta
    SigLIP tensors, preserving float32, while CLIPModel's `pos_embed` is created via
    `torch.from_numpy()` which also bypasses the meta context, staying float32. Adding
    float32 `pos_embed` to bfloat16 encoder output promotes to float32, causing a
    `RuntimeError: expected m1 and m2 to have the same dtype` in `attn_pool` — fixed by
    casting the entire `model.model.visual` module to the checkpoint dtype after loading.

### 2. Silicon SDPA hang (unfixed — compiler-stack bug)

After all loader fixes, CPU inference passes without error. The test then
proceeds to the silicon run, compiles the model to TT ops (StableHLO →
tt-mlir, ~1:40 minutes), dispatches the SigLIP `patch_embedding` Conv2D
successfully, and then dispatches the first SigLIP encoder SDPA operation:

```
2026-04-28 02:42:48.350 | info | Op | Multicast eligibility: 0/1 chains using mcast (all-or-nothing) (sdpa_program_factory.cpp:1272)
```

After this line, no further output is produced. The process enters a
`futex_wait_queue` state (all 250 threads blocked) with ~300% CPU from
UMD/host-side polling. `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` does not
trigger. The hang reproduces across multiple independent runs and persists
for more than one hour.

**SDPA configuration at hang:**
- Model: SigLIP ViT-L (SiglipVisionModel, 24 encoder layers)
- Image: 384×384, patch size 16 → 576 patches + 1 CLS = 577 tokens
- `num_attention_heads=16`, `hidden_size=1024` → head_dim=64
- Multicast: disabled (`0/1 chains using mcast`)

**Proposed fix:**
Investigate the SDPA kernel path triggered when `multicast=False` for this
shape on Blackhole hardware. Candidate hypotheses:
- The `mcast=all-or-nothing` SDPA path in `sdpa_program_factory.cpp` contains
  a kernel dispatch bug for seq_len=577 on Blackhole.
- `TT_METAL_OPERATION_TIMEOUT_SECONDS` is not plumbed through to the SDPA
  kernel dispatch / wait path, allowing hardware deadlocks to run unbounded.

Fix should live in **tt-metal** (`sdpa_program_factory.cpp` and/or the
timeout enforcement path for SDPA kernels).

## Fix
The loader-layer fixes are in `chexagent/pytorch/loader.py` in tt-forge-models.
They address eleven transformers 5.x incompatibilities. None of the fixes is
a forbidden workaround: the full 3B model is loaded, all SigLIP and LLM
weights are used, no layers are trimmed or offloaded to CPU, and all input
shapes are unchanged.

## Verification
CPU inference: PASS (no dtype errors, no segfault, visual encoder produces
output, LLM forward pass completes).

Silicon run: hangs at SDPA, never returns — pytest never exits. No SILICON_PASS
achieved; no wall-clock time to record.

Hardware: Blackhole (single chip, confirmed via `topology_discovery.cpp`
architecture log).

## Files changed
- `tt-xla/third_party/tt_forge_models/chexagent/pytorch/loader.py`
  — all eleven loader-layer fixes described above

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | fe54df8a100770139eed9587bb83a63506bbe472 |
| tt-forge-models | aac69782b2d05ed262752c3e443c70109833a775 |
