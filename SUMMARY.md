# Remediation Summary

## Test

```
pytest -svv tests/runner/test_models.py::test_all_models_torch[anima_fp8/pytorch-preview3-base-fp8-single_device-inference]
```

Expected failure: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

## Root Cause

The test machine has 32 × 1 GB hugepages configured in the kernel but `/dev/hugepages-1G` is not mounted as a hugetlbfs filesystem. UMD's `find_hugepage_dir()` scans `/proc/mounts` for that mount point and returns empty when it is absent. `SiliconSysmemManager::init_hugepages()` then silently returns false, leaving `hugepage_mapping_per_channel` empty. Later, `LocalChip::get_host_channel_size()` calls `TT_ASSERT(channel < get_num_host_channels())` which fails (0 channels available), throwing a `std::runtime_error`.

That exception propagated out of `tt::runtime::getCurrentSystemDesc()` inside the PJRT plugin. Because `populateDevices()` had no try-catch, the exception crossed the C/PJRT ABI boundary (undefined behavior) and bypassed the normal PJRT error-propagation path that would have produced a clean Python `RuntimeError`.

A second problem compounded the first: after the failed XLA client initialization torch_xla sets an internal `g_computation_client_initialized` flag. Any subsequent call to `GetComputationClient()` — including `xr.global_runtime_device_attributes()` inside `get_xla_device_arch()` in the `finally` block — triggers `CHECK(!g_computation_client_initialized)` which calls `abort()`. This turned a catchable test failure into an uncatchable process abort.

## Fixes

All changes are in `tt-xla` on branch `nsmith/fix-pjrt-exception-handling-and-arch-probe`.

### 1. `pjrt_implementation/src/api/client_instance.cc` — catch exceptions in `populateDevices()`

Wrapped `tt::runtime::SystemDesc::loadFromPath()`, `tt::runtime::getCurrentSystemDesc()`, and `getOrCreateMeshDevice()` in try-catch blocks. On exception the function logs the error and returns `tt_pjrt_status::kInternal`. This keeps the error inside the PJRT protocol so XLA can convert it to the expected Python `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

Commit: `3b9e7f1a9` — *pjrt: catch C++ exceptions in populateDevices() and return kInternal*

### 2. `tests/runner/test_utils.py` — guard `get_xla_device_arch()` against abort

Added module-level `_xla_device_arch_failed` and `_xla_device_arch_cache` flags. On the first call that succeeds the result is cached; on the first call that fails the flag is set and all subsequent calls return `""` immediately without touching `xr.global_runtime_device_attributes()`. This prevents the abort in the `finally` block.

Commit: `d8e55031e` — *tests: guard get_xla_device_arch() against second XLA client init*

## Result

After both fixes the test fails cleanly with the expected error:

```
FAILED tests/runner/test_models.py::test_all_models_torch[anima_fp8/pytorch-preview3-base-fp8-single_device-inference]
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## configure.sh fix

An unrelated typo was also found and fixed in `configure.sh`: `third_party/tt_forge_modules` → `third_party/tt_forge_models`. Without this fix the configure step fails because the directory does not exist. This fix is in the top-level repo on branch `test` (committed prior to this session).
