# AGENTS.md

Guidance for coding agents working in this KTransformers checkout.

Claude Code also loads [CLAUDE.md](CLAUDE.md). Keep the two files in sync when editing shared notes.

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

## Local mmap / cgroup memory-gradient testing

This is the procedure used on this host to measure DeepSeek-V2-Lite with **GPU non-expert / CPU routed-expert** and Linux mmap page cache under a shrinking cgroup memory cap. Scripts and completed runs live outside the repo:

`/tmp/kt-memory-breakdown-20260912-gpu1/`

Do not delete those results. Do not install KT into conda env `bitmoe`. Do not mix `cpu.max` / `io.max` into a memory-only experiment. Do not use `ulimit` as a substitute for cgroup.

### What is actually being tested

KTransformers has **no SSD expert pager** (no LRU that keeps only active experts resident and leaves the rest on disk). Closest production path:

- **NativeMoE BF16**: memcpy of experts into anonymous DRAM at load.
- **LLAMAFILE + GGUF `np.memmap`**: single-NUMA (`--kt-threadpool-count 1`) **aliases** the GGUF mmap (`m_local_*_proj_`); `MADV_WILLNEED` warms pages. Multi-NUMA copies slices and cannot alias.
- File-backed pages live in the kernel page cache (`RssFile` / cgroup `memory.stat.file`). There is no KT knob for cache size; the cgroup `memory.high` / `memory.max` cap is the **whole** group (anon + file + kernel), not a GGUF-only budget.
- In production `forward_one`, page-ins happen **inside** `llamafile_sgemm` (fused load+compute). They are not a separate load stage.

DeepSeek-V2-Lite shape used here: 27 layers, `first_k_dense_replace=1` → **26 MoE layers**, 64 experts, top-6, H=2048, I=1408, F16 ≈ 17.3 MB/expert, ≈ 104 MB/layer for 6 active experts.

### Environment

| Item | Value |
|---|---|
| Conda env | `ktransformers` at `/home/tzj/anaconda3/envs/ktransformers` (Python 3.12, `torch==2.9.1+cu128`, `kt-kernel` AVX2, `sglang-kt`) |
| Do not touch | conda `bitmoe` (torch 2.6.0+cu124, transformers 4.52.3, accelerate 1.7.0) or BitMoE repo pins |
| HF weights | `/home/tzj/models/DeepSeek-V2-Lite` (attention / shared) |
| CPU experts | `/home/tzj/gguf/DeepSeek-V2-Lite/DeepSeek-V2-Lite-F16.gguf` |
| GPU | physical GPU1, UUID `GPU-480c4b41-8579-8b74-a88e-e1aa67851033` via `CUDA_VISIBLE_DEVICES=<UUID>` then NVML check |
| Disk | Intel SSDPF2KX153T1 NVMe, sequential O_DIRECT ≈ 4.6 GB/s (order-of-magnitude only) |

Always set `SGLANG_APPLY_CONFIG_BACKUP=none`. Otherwise SGLang maps 27-layer V2 onto a V4-small backup config.

`--proc/environ` may be empty because of `setproctitle`. Do not use it as the sole GPU-binding proof; check NVML (`nvidia-smi --query-compute-apps=pid,gpu_uuid`).

### Local source enablement (not upstream)

LLAMAFILE refuses `intermediate_size` not divisible by `QK_K=256`. That check is only valid when **slicing** FFN across TP partitions. DeepSeek-V2-Lite has I=1408; with `--kt-threadpool-count 1` the partition owns the full width and aliases mmap.

Keep this uncommitted working-tree change (do not revert it if you still need to launch LLAMAFILE on this model):

- `kt-kernel/operators/moe-tp.hpp`: `if (is_llamafile && tp_count > 1)` before the `% QK_K` throw
- `kt-kernel/python/utils/llamafile.py`: raise only when `threadpool_count > 1`

This is **not** a `1408` special case. `tp_count==1` falls through to equal division (`intermediate_size /= 1`).

Decode-phase wall-clock timing is already in HEAD (`KT_MOE_PHASE_TIMING` in `kt-kernel/operators/llamafile/moe.hpp`). Env unset → one `getenv` + branch. Default print interval 4096; the bench sets `KT_MOE_PHASE_TIMING_INTERVAL=26`.

The pretouch diagnostic (`KT_PROFILE_EXPERT_IO` serial 4 KiB touch + `getrusage` + `[KT_IO]` / `[KT_IO_FAULTS]`) was **removed from the production hot path**. Re-apply only when reproducing pretouch:

```bash
cd /home/tzj/Code/ktransformers
git apply /tmp/kt-memory-breakdown-20260912-gpu1/expert-io-probe.patch
# rebuild kt-kernel in env ktransformers, then run bench.py --mode pretouch
```

See `/tmp/kt-memory-breakdown-20260912-gpu1/REPRO-PRETOUCH.txt`. Native mode does **not** need that patch. Do not commit the probe.

### Smoke inference (no cgroup)

GPU attention/shared, CPU routed experts, single NUMA, no GPU routed experts:

```bash
conda activate ktransformers
export CUDA_VISIBLE_DEVICES=GPU-480c4b41-8579-8b74-a88e-e1aa67851033
export SGLANG_APPLY_CONFIG_BACKUP=none
python -m sglang.launch_server \
  --host 127.0.0.1 --port 30124 \
  --model /home/tzj/models/DeepSeek-V2-Lite \
  --kt-weight-path /home/tzj/gguf/DeepSeek-V2-Lite/DeepSeek-V2-Lite-F16.gguf \
  --kt-method LLAMAFILE \
  --kt-cpuinfer 32 --kt-threadpool-count 1 \
  --kt-num-gpu-experts 0 --kt-max-deferred-experts-per-token 0 \
  --attention-backend triton --trust-remote-code \
  --mem-fraction-static 0.08 --chunked-prefill-size 256 \
  --max-running-requests 1 --max-total-tokens 512 --context-length 1024 \
  --tensor-parallel-size 1 --disable-overlap-schedule --disable-cuda-graph \
  --cuda-graph-max-bs 1 --random-seed 42 --skip-server-warmup
```

`--mem-fraction-static 0.7` OOMs the KV pool on A100 40GB after experts are placed; 0.08 + 512 total tokens is the working serving envelope for this setup.

### cgroup v2 protocol

Use **cgroup v2 memory controller**, not `ulimit`. Launch the server as a child of a user scope so it inherits the group (you cannot move PIDs from a session cgroup into `user@` after the fact):

```bash
systemd-run --user --scope --unit=kt-mem-exp.scope \
  --property=MemoryHigh=48G \
  --property=MemoryMax=64G \
  --property=MemorySwapMax=0 \
  --working-directory=/tmp -- \
  /home/tzj/anaconda3/envs/ktransformers/bin/python -u -m sglang.launch_server ...
```

Pitfalls already hit on this machine:

- Write `--property=MemoryHigh=48G`. `-p=MemoryHigh=48G` is parsed as a property named `=MemoryHigh`.
- Inspect `memory.high` / `memory.max` / `memory.swap.max` / `cgroup.procs` **while the scope is alive**. After exit, `systemctl --user show` can look unlimited.
- `MemoryOOMGroup=yes` is unknown on systemd 249; do not set it.
- Cannot `drop_caches` without sudo. Evict **only** the GGUF with `posix_fadvise(..., POSIX_FADV_DONTNEED)` and verify with `mincore`. Abort the case if remaining resident pages > 1%.
- `MemoryHigh=2G` is below ~2.7–3.5 GiB anonymous and the server never becomes ready. Formal ladder starts at 4 GiB high.
- Do not add `cpu.max` or `io.max` to a memory-gradient run.
- `SO_REUSEADDR` on the probe bind; port 30124 can sit in `TIME_WAIT`.

Formal ladder (`high` / `max` GiB): **48/64, 16/20, 8/12, 4/8**, all with `memory.swap.max=0`.

### Two measurement modes

Driver: `/tmp/kt-memory-breakdown-20260912-gpu1/bench.py`.

Workload: one fixed prompt (~20 tokens), `temperature=0`, **32** completion tokens, streaming HTTP for E2E and TTFT. Per case: **2 warmup + 2 measured** requests. Repeated prompts hit SGLang prefix cache; numbers are for this short decode workload, not arbitrary long prefill. First warmup prefill goes through `forward_many` and is **not** covered by `forward_one` phase timers — do not treat warmup as a complete expert-stage breakdown.

**native** (throughput / attribution under real fused load+compute):

- No `KT_PROFILE_EXPERT_IO`.
- `KT_MOE_PHASE_TIMING=1`, interval 26.
- Stages: input quant; gate/up + activation + down input convert; down + weighted accumulate; TP merge.
- These walls **include** page-ins, reclaim, and `memory.high` throttle. They are not pure GEMM.

**pretouch** (diagnostic only, 3–4× slower under low memory):

- Requires `expert-io-probe.patch` applied and kt-kernel rebuilt.
- Before GEMM, single-threaded 4 KiB-stride touch of each active expert’s gate/up/down, then the original compute stages.
- Records `load_ns` / `compute_ns` plus process-wide maj/min fault and user/sys CPU seconds.
- Changes fault timing and I/O concurrency. Do **not** quote pretouch E2E as mmap throughput.
- Touch time includes DRAM access, faults, reclaim, throttle — not “pure SSD”. If `compute_maj > 0`, post-touch compute still waited on disk.

For every **measured** request, `forward_one` timer calls must be `26 × 32 = 832`. Otherwise reject the breakdown.

### How to run

```bash
conda activate ktransformers
python /tmp/kt-memory-breakdown-20260912-gpu1/bench.py --mode native
# pretouch only after git apply expert-io-probe.patch and rebuild:
python /tmp/kt-memory-breakdown-20260912-gpu1/bench.py --mode pretouch
python /tmp/kt-memory-breakdown-20260912-gpu1/analyze.py
python /tmp/kt-memory-breakdown-20260912-gpu1/validate.py
```

`bench.py` creates a fresh scope per case, refuses to reuse a live unit name, writes `native-h*/` or `pretouch-h*/` (`server.out`, `samples.jsonl`, `result.json`), and stops the scope it created even on failure.

### Validation gates (`validate.py`)

A case is only a performance sample if all of these hold:

- `ok`, no sampler errors, 2 warmups + 2 measured runs
- `cache_initial.after.resident_pages == 0` (mincore after DONTNEED)
- `scope_after_stop == inactive`
- NVML GPU UUID matches the intended device for every worker in the group
- each measured request: `completion_tokens == 32`, phase `calls == 832`
- PID set stable across the request; `memory.high` / `max` match the configured GiB; `swap.max == 0` and `swap.current == 0`; `oom_kill == 0`
- `quant+gateup+down+merge <= e2e + 5ms`
- pretouch: faults `n == 832`, `load_ns >= 0`, `compute_ns > 0`
- all measured completions share one `text_sha256`

### How to read the numbers

Do not add overlapping clocks. Do not treat process CPU seconds as wall time.

| Signal | Meaning | Not |
|---|---|---|
| `[KT_PHASE]` quant/gateup/down | wall time of that `forward_one` region | pure GEMM or pure SSD |
| `[KT_PHASE merge]` | TP merge wall | GPU time |
| `E2E − expert stages − merge` | residual (GPU non-expert, copy, sample, HTTP, …) | “pure GPU” |
| `read_bytes` | all files read by processes in the group | expert-logical bytes |
| `pgmajfault`, `workingset_refault_file` | cgroup-wide file faults | expert cache miss rate |
| `memory.events.high` | times the high watermark fired | duration |
| PSI `some` / `full` for io, memory, cpu | stall windows; they overlap | addable fractions of E2E |
| cgroup/process CPU seconds | sum over threads | wall-clock share |
| pretouch `load_ns` | serial touch wall | SSD bandwidth time |
| `file` vs `file_mapped` | `file_mapped` is a subset of `file` | do not add them |
| lifetime `[KT_PHASE]` / `[KT_IO]` percents on stderr | process lifetime averages | per-request deltas (always subtract request-before/after) |

Completed 8-way results (do not overwrite): `native-h{48,16,8,4}/` and `pretouch-h{48,16,8,4}/`, plus `aggregated.json`, `validation.json`, `METHODS.md`, `REPORT.md`.

### Explicitly out of scope for this bench

- Committing probes, QK_K enablement, or `/tmp` scripts into the repo unless the owner asks
- Rebuilding or re-running GPU cases as a drive-by after doc-only edits
- Changing BitMoE pins or conda env `bitmoe`
- Using NativeMoE as a stand-in for mmap (it copies experts into anon DRAM)
- Claiming KT “SSD offload of inactive experts” beyond page-cache eviction under memory pressure
