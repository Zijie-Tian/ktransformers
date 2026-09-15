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

## Jetson runbook: agx / edge1 / edge2 / edge3

Inventory and deployed launcher paths were checked over SSH on **2026-09-15**. The successful generation dates below refer to saved end-to-end tests, **not a new inference run during this documentation update**.

| SSH host | Hardware / system | CUDA used by `ktransformers` | Last successful full-model test |
|---|---|---|---|
| `agx` | AGX Orin, SM87; Ubuntu 22.04, L4T 36.3.0 | System `/usr/local/cuda-12.2` | 2026-09-12: Llama-3.1-8B-Instruct, FP16 / Triton, Paris and `5.` |
| `edge1` | AGX Xavier, SM72; Ubuntu 20.04, L4T 35.6.4 | Private CUDA 12.2; system CUDA 11.4 preserved | 2026-09-13: Llama-3.2-1B-Instruct, FP16 / `torch_native`, Paris and five |
| `edge2` | AGX Xavier, SM72; Ubuntu 20.04, L4T 35.6.4 | Same private CUDA profile | 2026-09-13: same two Llama-1B answers |
| `edge3` | AGX Xavier, SM72; Ubuntu 20.04, L4T 35.6.4 | Same private CUDA profile | 2026-09-13: same two Llama-1B answers |

These are **dense CUDA inference** results through `kt run` → SGLang-KT. They do not demonstrate CPU routed-expert offload, arbitrary model support, quantized inference, or multi-GPU/multi-host serving. Edge1 received additional experimental DeepSeek changes after its dense baseline; rerun the relevant smoke before accepting a changed deployment.

### Environment and deployment boundaries

- All four SSH aliases use user `edgellm`, home `/home/edgellm`, checkout `~/Code/ktransformers`, and conda `~/anaconda3/envs/ktransformers` (Python 3.11.16). Do not install into `bitmoe` or change system JetPack/driver/CUDA.
- Verified package pins: source-built Torch **2.9.1 / CUDA 12.2**; `ktransformers`, `kt-kernel`, `sglang-kt` **0.7.0.post4**; `sgl-kernel` **0.3.21**; `transformers-kt` **5.6.0.post3**; Triton **3.5.1**; torchvision **0.24.1**; torchaudio **2.9.1**. Native packages were built for the target Jetson ABI. Never copy Orin SM87 binaries to Xavier SM72, or workstation x86 build outputs to either.
- Communication uses the local opt-in **`SGLANG_JETSON_SINGLE_GPU_GLOO=1`**, restricted to world size 1; model computation remains on CUDA. On AGX, NCCL compiled but failed the Jetson NVML P2P runtime query. On edges, NCCL is not compiled. Neither is a validated NCCL serving configuration. **`USE_LIBUV=0`** selects the classic TCPStore required by these Torch builds.
- Keep **`SGLANG_APPLY_CONFIG_BACKUP=none`** and the environment's **`PIP_CONSTRAINT`**. Installing vanilla `transformers` would overwrite the KT fork's namespace; unpinned dependency upgrades can replace the native Torch stack. The installed constraints and metadata-adjusted dependency wheels are recorded in each repair directory.
- HF checkpoints belong under `~/models/`; GGUF weights under `~/gguf/<ModelName>/`. The edge Llama-1B HF directory was exported from the existing GGUF. Do **not** substitute edge1's incomplete `~/models/Llama-3.1-8B-Instruct` directory.
- These commands target the **already provisioned devices**, not a clean upstream checkout. On the inventory date, all four deployed branches were `tzj/bitmoe` at `440df94` with local compatibility patches, including changes inside `third_party/sglang` (pinned base `3424f35`). The control checkout has newer commits. Preserve and reconcile remote patches before updating; do not `reset --hard`, force-update submodules, or overwrite remote trees with workstation binaries.
- `.omx/setup-agx/` and `.omx/setup-edge/` contain **ignored, device-local** launchers, wheels/patch manifests, environment snapshots and test evidence. They are not included by `git clone` or by this documentation commit. Recover the corresponding device artifacts before using these instructions on a fresh installation; the upstream SM75 policy has not been globally disabled for Xavier.

### Start the existing validated dense profiles

From the control machine, choose one command (foreground; Ctrl-C to stop your server):

```bash
ssh agx 'bash ~/Code/ktransformers/.omx/setup-agx/inference-repair-20260912/serve-llama.sh'
ssh edge1 'bash ~/Code/ktransformers/.omx/setup-edge/serve-llama.sh'
ssh edge2 'bash ~/Code/ktransformers/.omx/setup-edge/serve-llama.sh'
ssh edge3 'bash ~/Code/ktransformers/.omx/setup-edge/serve-llama.sh'
```

Each launcher activates the correct environment and binds **`127.0.0.1:30124`**. Check that the port is unused; do not stop someone else's service. For client access from the control machine, use a separate tunnel such as `ssh -N -L 30124:127.0.0.1:30124 edge1` with a free local port, rather than exposing the server on `0.0.0.0`.

The equivalent commands below explain the required settings. Run them **on the selected device in a clean Bash shell**, not on the control machine. First activate the shared environment:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ktransformers
cd ~/Code/ktransformers
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_APPLY_CONFIG_BACKUP=none SGLANG_JETSON_SINGLE_GPU_GLOO=1 USE_LIBUV=0
```

**AGX Orin only:** conda activation selects the system CUDA 12.2 runtime. Use the existing complete Llama-8B checkpoint and Triton attention:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.2/bin/ptxas
kt run "$HOME/models/Llama-3.1-8B-Instruct" \
  --weights-path "$HOME/gguf/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-F16.gguf" \
  --kt-method LLAMAFILE --host 127.0.0.1 --port 30124 \
  --served-model-name agx-kt-smoke-20260912 \
  --tensor-parallel-size 1 --gpu-experts 0 --cpu-threads 4 --numa-nodes 1 \
  --attention-backend triton --dtype half \
  --max-total-tokens 512 --max-running-requests 1 --chunked-prefill-size 64 \
  --mem-fraction-static 0.55 --context-length 1024 \
  --disable-cuda-graph --disable-overlap-schedule --disable-custom-all-reduce \
  --random-seed 42 --skip-server-warmup
```

**edge1 / edge2 / edge3 Xavier only:** source the private runtime helper after conda activation. It sets `CUDA_HOME=~/opt/ktransformers-cuda-12.2/root/usr/local/cuda-12.2`, puts its `compat/` and `lib64/` ahead of system libraries, and retains the JetPack CUDA 11.4/cuDNN/Tegra library paths. Do not replace the system driver or change global CUDA symlinks.

```bash
source .omx/setup-edge/runtime-env.sh
export SGLANG_JETSON_SM72_DENSE=1
export NVCC_PREPEND_FLAGS="--compiler-bindir=$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TVM_FFI_CACHE_DIR="$HOME/.cache/ktransformers-sm72-gcc11-20260913/tvm-ffi"
kt run "$HOME/models/Llama-3.2-1B-Instruct-from-GGUF" \
  --weights-path "$HOME/gguf/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-f16.gguf" \
  --kt-method LLAMAFILE --host 127.0.0.1 --port 30124 \
  --served-model-name xavier-kt-smoke-20260913 \
  --tensor-parallel-size 1 --gpu-experts 0 --cpu-threads 4 --numa-nodes 1 \
  --attention-backend torch_native --sampling-backend pytorch --dtype half \
  --max-total-tokens 256 --max-running-requests 1 --chunked-prefill-size 64 \
  --mem-fraction-static 0.40 --context-length 1024 \
  --disable-cuda-graph --disable-overlap-schedule --disable-custom-all-reduce \
  --random-seed 42 --skip-server-warmup
```

The explicit conda **GCC 11.4** selection is necessary for SGLang's C++20 CUDA JIT; the edges' system GCC 9 fails on `<concepts>`. `CXX` alone did not select NVCC's host compiler. The SM72 dense mode is deliberately restricted to single-GPU, unquantized FP16 Llama, `torch_native` CUDA attention and PyTorch sampling; a basic Triton probe passing does not establish general Triton attention support on Xavier.

### Smoke and acceptance evidence

In another terminal on that device (or through the SSH tunnel), wait for the expected model ID in `/v1/models`, then submit an actual generation request:

```bash
curl -fsS --max-time 10 http://127.0.0.1:30124/v1/models
MODEL=xavier-kt-smoke-20260913  # On AGX: MODEL=agx-kt-smoke-20260912
curl -fsS --max-time 180 http://127.0.0.1:30124/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Answer in one short sentence: What is the capital of France?\"}],\"temperature\":0,\"max_tokens\":32,\"stream\":false}"
```

Require a nonempty answer containing **Paris**; repeat with `Answer briefly: What is two plus three?` and require **5/five**. Imports, `/v1/models`, a ready log, or an isolated native MoE test are not end-to-end inference acceptance. The first request may include tens of seconds of JIT compilation; do not report that as steady-state throughput. Stop only the server/process group you started, and verify its children and port are gone; parent exit alone is insufficient.

Saved successful evidence (paths relative to the corresponding device checkout):

- AGX: `.omx/setup-agx/inference-repair-20260912/final-validation.json` and `generation-attempt2/result.json` in that directory. Actual answers: `The capital of France is Paris.` and `5.`.
- Each edge: `.omx/setup-edge/inference-repair-20260913/final-validation.json` and `generation-sm72-native2/`. Actual answers: `Paris is the capital of France.` and `Two plus three equals five.`. The control checkout also retains `fleet-final-validation.json` under the same repair directory.
- Each repair directory retains `serve-llama.sh` or launcher-validation evidence, the generation driver, `environment-final.yml`, dependency snapshots and native/source hashes. Use a **fresh attempt directory** for retesting. After source or binary changes, rerun the relevant native/serving gates; do not edit old readiness hashes to make stale evidence pass.
- The moved WorkerPool regression is now local-only at `.omx/setup-edge/regression-tests/test_worker_pool_config.py`; older remote gates still reference their original test path. If deploying updated validation helpers, also deploy the matching test and regenerate gates by actually running them. Do not mix new helpers with old hash markers.

### DeepSeek-V2-Lite SSD-backed experts: not working yet

Do not present the dense results above as DeepSeek success. As of this inventory:

- All three edges have the previously SHA256/tensor-index-verified HF checkpoint at **`~/models/DeepSeek-V2-Lite`** (4 safetensors shards; 31,418,836,616 bytes across the copied files). Copy evidence is in `.omx/setup-edge/deepseek-v2-lite-copy-20260913/` on the control machine.
- **Only edge1** has `~/gguf/DeepSeek-V2-Lite/DeepSeek-V2-Lite-Experts-F16.gguf`: **28,789,709,120 bytes**, SHA256 `d25bd3ae5f00268696305726b936dbea5ee3753c0f9ddc18919266b448c0bb2c`. This is expert-only F16 GGUF; tokenizer, attention, shared and other non-expert weights still come from HF. Edge2/3 do not have this GGUF or a tested DeepSeek serving profile.
- Edge1's 2026-09-14 attempt loaded the model and verified **78 native expert pointers aliasing the SSD GGUF**, then aborted on the first generation in ARM F16 expert matvec: **`std::runtime_error: llamafile not supported` / SIGABRT**. A CPU-only H2048/I1408 reproduction also failed. **Zero completed text responses; no F16 fix applied yet.** Native FP32 MoE and GPU attention/router numerical passes do not resolve this failure.
- The experimental launcher and logs are under `.omx/setup-edge/deepseek-ssd-inference-20260914/` on edge1 and the control machine. `serve-deepseek.sh` targets localhost **30125**, but it is a **debugging profile, not a working inference command**. It requires the separate guarded `SGLANG_JETSON_SM72_DEEPSEEK=1` path, Triton MLA tile changes, and `SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch` for correct GPU FP32 router accumulation with FP16 inputs. Do not substitute the dense SM72 flag or roll it out as a validated edge2/3 configuration.
- KT has **no independent SSD expert pager/LRU**. The intended path is **LLAMAFILE + GGUF mmap + one threadpool**, with zero GPU routed experts and GPU expert prefill disabled. Linux manages file-backed page cache; NativeMoE BF16 copies experts into anonymous DRAM and is not equivalent. No bounded-cache or memory-capped DeepSeek throughput result has passed on these devices. Jetson memory cgroups were v1/hybrid, so the separate workstation cgroup-v2 recipe below is not directly applicable.

Next acceptance order remains: fix and numerically test ARM F16 single-/multi-token expert execution while preserving mmap, obtain real edge1 generation, then deploy and independently verify edge2 and edge3.

### Repair / rebuild precautions

1. Preserve the tested core backups: `ktransformers-core-backup-20260912` on AGX; `ktransformers-core-backup-20260913` on edges. These are **core-only rollback environments**, not replacement serving environments. Never rerun the older core-only Torch build script over the serving environment: it disables distributed support required by SGLang.
2. Start from each device's `inference-repair-20260912/` (AGX) or `inference-repair-20260913/` (edge) scripts, source pins, constraints and wheel manifests. Rebuild Torch/native extensions for the actual SM and ABI; share wheels between edges only after ABI/hash checks and numerical tests on the recipient. Preserve failed attempts and source patches rather than repeatedly reinstalling arbitrary packages.
3. Xavier's NUMA-disabled kernel needs the WorkerPool logical-node-0 / hwloc allowed-CPU-set fallback (`b71c2bc` in this branch). The LLAMAFILE single-partition QK_K guard fix is `a7cda38`. Device SGLang patches additionally handle guarded Gloo, native architecture exclusions, and on Xavier the real CUDA memory query when `nvidia-smi` is absent. A missing `nvidia-smi` executable or an x86-centric `kt doctor` label is not proof that CUDA is unusable; use actual device/ELF and numerical checks, never fake command output.
4. Validate real CPU/CUDA Gloo rendezvous, core CUDA/cuDNN/native kernels, `python -m pip check`, then actual generated text for the selected model/profile. Preserve default architecture restrictions and explicit errors for unsupported kernels. Do not infer BF16/FP8/FP4, SFT, all-model or multi-host support from the FP16 dense smoke.

## Local mmap / cgroup memory-gradient testing

This is a separate workstation experiment, **not** the Jetson operating procedure above. Its absolute paths, GPU UUID and cgroup-v2 settings must not be copied to the Jetsons unchanged.

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

This branch includes the single-partition guard fix (`a7cda38`); preserve it when updating or deploying:

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

- Committing additional probes or `/tmp` scripts into the repo unless the owner asks
- Rebuilding or re-running GPU cases as a drive-by after doc-only edits
- Changing BitMoE pins or conda env `bitmoe`
- Using NativeMoE as a stand-in for mmap (it copies experts into anon DRAM)
- Claiming KT “SSD offload of inactive experts” beyond page-cache eviction under memory pressure
