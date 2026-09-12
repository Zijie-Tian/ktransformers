# CLAUDE.md

Project guidance for Claude Code. Other coding agents should follow [AGENTS.md](AGENTS.md).

## `third_party/llama.cpp` looks dirty after a build

This is expected. Do not commit it, and do not treat it as unrelated local edits.

The submodule is pinned to upstream `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` (llama.cpp tag `b3173`). That commit predates MXFP4 and a NumPy 2 `gguf-py` byte-order fix, so `kt-kernel/CMakeLists.txt` applies this patch series at configure time, immediately before `add_subdirectory(llama.cpp)`:

- `kt-kernel/third_party_patches/llama.cpp/0001-ggml-mxfp4-type.patch` — `GGML_TYPE_MXFP4` for DeepSeek-V4 CPU experts
- `kt-kernel/third_party_patches/llama.cpp/0002-gguf-numpy2-byteorder.patch` — NumPy 2 `newbyteorder` in `GGUFReader`

The patches are applied **in place** on the submodule working tree. After a successful configure you will see `m third_party/llama.cpp` in the superproject, and `git -C third_party/llama.cpp status` will list `ggml*.h` / `ggml-quants.*` / `gguf-py/...`. The gitlink SHA is unchanged.

Rules:

1. Do **not** `git add third_party/llama.cpp` or commit a new submodule SHA for these files.
2. Do **not** reset the submodule as a drive-by cleanup (`git submodule update --force`, `git checkout -- third_party/llama.cpp`). The next `cmake` / `./install.sh` will apply the same patches again.
3. To restore a clean pin without configuring: reverse-apply the patches, newest first. See `kt-kernel/third_party_patches/llama.cpp/README.md`.
4. CMake is idempotent and uses no marker file inside the submodule (a marker would show up as untracked forever): `git apply --check`, else `git apply --reverse --check` meaning “already applied”.
5. If `git apply --reverse --check` fails for those two patches, the submodule has **extra** local edits on top of the official series — stop and inspect; do not blindly discard them.
