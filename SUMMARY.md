# Remediation Summary: llava_onevision_1_5-pytorch-4B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llava_onevision_1_5/pytorch-4B_Instruct-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: visual encoder rot_pos_emb and cu_seqlens paths require device-to-host scalar transfer inside compiled XLA graph, which the TT PJRT backend does not support (INTERNAL: Error code: 13)

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Full traceback (innermost relevant frames):
  modeling_llavaonevision1_5.py:1025 — rotary_pos_emb = self.rot_pos_emb(grid_thw)
  loader.py:98                        — rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
  modeling_llavaonevision1_5.py:223   — def forward(self, seqlen: int) -> torch.Tensor:
  torch_xla/_dynamo/dynamo_bridge.py:826 — torch_xla.sync(reset_scope=False)
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Preceding graph break warnings (multiple):
  Backend compiler `tt` failed with aten._local_scalar_dense.default
  Explanation: rot_pos_emb requires tolist() / .item() on TT device tensors inside traced graph

## Root cause
Four loader-layer bugs blocked the model from loading under transformers 5.x. All four were fixed in the loader:

1. **SlidingWindowCache removed** — transformers 5.x no longer exports SlidingWindowCache. The model's trust_remote_code module imports it at module level. Fix: inject a no-op stub class into transformers.cache_utils.

2. **use_fast default changed** — In transformers 5.x, AutoProcessor.from_pretrained loads Qwen2VLImageProcessor as fast processor by default. Fix: pass use_fast=False.

3. **pad_token_id absent from sub-config** — LLaVAOneVision1_5_TextModel.__init__ reads config.pad_token_id; absent in transformers 5.x sub-configs. Fix: add PretrainedConfig.pad_token_id = None stub.

4. **'default' missing from ROPE_INIT_FUNCTIONS** — transformers 5.x removed the 'default' key; the model's custom rotary embedding does ROPE_INIT_FUNCTIONS[self.rope_type] with rope_type='default'. Fix: inject a standard RoPE inv_freq implementation under that key.

After fixing all four loader bugs, compilation fails with INTERNAL: Error code: 13 in the visual encoder. The rot_pos_emb path calls grid_thw.tolist() but Dynamo traces grid_thw as a device tensor in the outer compilation context, triggering aten._local_scalar_dense (device-to-host scalar extraction) which the TT PJRT backend does not support inside compiled graphs. This is the same Tier B pjrt-device-to-host-transfer bug reported for the sibling 4B_Base and 8B_Instruct variants.

## Fix
Loader layer (committed, pushed to remediation branch in tt-forge-models):
- Add SlidingWindowCache stub to transformers.cache_utils
- Add PretrainedConfig.pad_token_id = None stub
- Add ROPE_INIT_FUNCTIONS["default"] stub with standard RoPE inv_freq formula
- Pass use_fast=False to AutoProcessor.from_pretrained
- Add _patch_model_for_tt() with rot_pos_emb, get_image_features, get_rope_index, and RiceSdpaAttention patches (adapted from causal_lm sibling branch; does not fully resolve the INTERNAL error)

Compiler-stack layer (Tier B, not attempted):
The fix would live in tt-xla. When lowering aten._local_scalar_dense or torch.arange with a device tensor as size argument inside a compiled XLA graph, the PJRT backend needs stablehlo.dynamic_iota support or device-to-host scalar constant folding. Neither path exists today.

## Tier B justification
Indicator: new-infrastructure

Supporting aten._local_scalar_dense inside compiled XLA graphs requires either implementing stablehlo.dynamic_iota lowering in the TT compiler pipeline, or implementing device-to-host scalar transfer for compile-time constant folding in the PJRT bridge. Neither path exists today.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    230.46s (0:03:50)
- Tier A attempts: N/A

## Files changed
- tt-forge-models/llava_onevision_1_5/pytorch/loader.py (1 commit on remediation/llava_onevision_1_5-pytorch-4B_Instruct-single_device-inference)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 40691c565306e2ad25a93fffb12a76c1f1197479 |
