Third-party vendored projects

This directory contains vendored source code from external projects required by this
project. Large weight files (>100MB) are intentionally excluded from the vendor
snapshot. See `third_party_weights_manifest.json` for a complete list of excluded
large files (paths, sizes, SHA256 checksums).

How to restore weights

1. If you have the original weights in your local tree (e.g., `vggt/weight/...`), run:

    bash ./scripts/fetch_weights.sh

   This will copy weights from their original locations into `third_party/<repo>/...`.

2. If you prefer to download weights from a remote location, add the files to the
   original path listed in `third_party_weights_manifest.json` and rerun the script.

Credits and sources

- croco: https://github.com/naver/croco
- dinov3: https://github.com/facebookresearch/dinov3
- vggt: https://github.com/facebookresearch/vggt
- Depth-Anything-3: https://github.com/ByteDance-Seed/Depth-Anything-3
- DP/diffusion_policy: https://github.com/your-org/diffusion_policy (vendored snapshot)

Licenses

Each vendored project retains its original license; refer to each project's
repository for license details.
