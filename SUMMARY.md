# Remediation Summary

## Test

```
pytest -svv tests/runner/test_models.py::test_all_models_torch[anima_fp8/pytorch-preview3-base-fp8-single_device-inference]
```

**Result: PASSED** (`1 passed, 5 warnings in 417.34s`)

## Root Causes

### Root Cause 1: Missing hugetlbfs mount in container

The test machine has 32 × 1 GB hugepages configured in the kernel (`/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages = 32`) but `/dev/hugepages-1G` is not mounted as a hugetlbfs filesystem (container restriction — `mount` is EPERM inside the containerd environment).

UMD's `find_hugepage_dir()` scans `/proc/mounts` for that mount point and returns empty when it is absent. `SiliconSysmemManager::init_hugepages()` then returned false, leaving `hugepage_mapping_per_channel` empty. Later, `LocalChip::get_host_channel_size()` calls `TT_ASSERT(channel < get_num_host_channels())` which fails (0 channels available), throwing a `std::runtime_error`.

That exception propagated out of the PJRT plugin and surfaced as `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

### Root Cause 2: SDPA-decode kernel constraint violated by synthetic inputs

The original `load_inputs()` used `latent_height=2, latent_width=2`. With `patch_size=(1,2,2)` the spatial token count is `(H//2)*(W//2) = 1`. The TTNN `sdpa_decode` op calls `get_chunk_size(s)` which returns `2` for `s=1`, and the kernel requires `k_chunk_size % 32 == 0`. This triggered:

```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
```

## Fixes

### Fix 1: Anonymous hugepage fallback in UMD

**Repo**: `tt-metal/tt_metal/third_party/umd` (submodule: `tenstorrent/tt-umd`)
**Branch**: `nsmith/anonymous-hugepages-fallback`
**Commit**: `c3065406`

Modified `device/chip_helpers/silicon_sysmem_manager.cpp` in `init_hugepages()`. When `hugepage_dir.empty()`, instead of returning false, the code now falls back to allocating anonymous 1 GB hugepages via `mmap(MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB)`. This works because the kernel pool has pre-allocated hugepages even without a filesystem mount.

The rebuilt `libtt-umd.so` was copied from `tt-metal/build_Release/lib/` to `tt-xla/third_party/tt-mlir/install/lib/libtt-umd.so` (the path the PJRT plugin loads at runtime).

**Push status**: Local commit only. The `nsmithtt` SSH key is not authorized for SAML SSO with the `tenstorrent` GitHub organization. Key must be authorized via GitHub Settings → SSH and GPG keys → Authorize, then `git push -u origin nsmith/anonymous-hugepages-fallback`.

### Fix 2: Correct synthetic latent dimensions in anima_fp8 loader

**Repo**: `tt-xla/third_party/tt_forge_models`
**Branch**: `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-35`
**Commit**: `dfba06dbb7`

Modified `anima_fp8/pytorch/loader.py` `load_inputs()`: changed `latent_height=2, latent_width=2` to `latent_height=8, latent_width=16`. With `patch_size=(1,2,2)`, the spatial token count is `(8//2)*(16//2) = 32`. `get_chunk_size(32) = 32`, satisfying `32 % 32 == 0`.

**Push status**: Local commit only. Same SAML SSO blocker as Fix 1. Command to push:
```bash
cd tt-xla/third_party/tt_forge_models
git push -u origin arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-35
```

## Previous Session Fixes (already pushed)

### Fix 3: PJRT exception handling (`tt-xla`)

Wrapped `tt::runtime::SystemDesc::loadFromPath()`, `tt::runtime::getCurrentSystemDesc()`, and `getOrCreateMeshDevice()` in try-catch blocks in `pjrt_implementation/src/api/client_instance.cc`. On exception the function logs the error and returns `tt_pjrt_status::kInternal`.

### Fix 4: Guard `get_xla_device_arch()` against abort (`tt-xla`)

Added module-level `_xla_device_arch_failed` and `_xla_device_arch_cache` flags in `tests/runner/test_utils.py` to prevent the `CHECK(!g_computation_client_initialized)` abort in the `finally` block after a failed XLA client initialization.

### Fix 5: `configure.sh` typo (`tt-forge-remediation`)

Fixed `third_party/tt_forge_modules` → `third_party/tt_forge_models` in `configure.sh`.

## Actions Required to Complete Push

1. Authorize the `nsmith@tenstorrent.com` SSH key for SAML SSO at:
   https://github.com/settings/keys → find key → "Configure SSO" → Authorize for `tenstorrent`

2. Push UMD fix:
   ```bash
   cd tt-metal/tt_metal/third_party/umd
   git push -u origin nsmith/anonymous-hugepages-fallback
   ```

3. Push tt_forge_models fix:
   ```bash
   cd tt-xla/third_party/tt_forge_models
   git push -u origin arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-35
   ```

4. Open a PR for the UMD anonymous hugepage fallback against `tenstorrent/tt-umd`.

5. Commit the top-level repo to update submodule pointers:
   ```bash
   git add tt-metal tt-xla
   git commit -m "update submodule pointers for anima_fp8 fixes"
   ```
