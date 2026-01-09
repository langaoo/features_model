Integration backups & notes

This directory contains backups produced during the vendor operation.

- `backups/<repo>/` contains:
  - `<repo>_changes.patch` - uncommitted modifications diff
  - `<repo>_staged_changes.patch` - staged modifications diff
  - `<repo>_untracked.tar.gz` - archive of untracked files
  - `<repo>_large_files.txt` - list of large (>100MB) files discovered
  - `<repo>_large_files.sha256` - SHA256 checksums for large files

How to restore

To apply patches:

  cd <repo>
  git apply ../../integration/backups/<repo>/<repo>_changes.patch

To extract untracked files:

  tar -xzf ../../integration/backups/<repo>/<repo>_untracked.tar.gz -C ./

Notes

- These backups are safe: they do not modify your original repositories.
- If you want to publish your modified sub-repos to your GitHub, create a fork
  and push the branch created for the integration (contact me to perform the
  push or provide remote URL and I will do it for you).
