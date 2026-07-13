# LLaMA-2 CUDA Inference — GPU Performance & Bottleneck Analysis

A complete CUDA C++ port of [Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c), extended with per-token profiling, NVML-based GPU power sampling, and a three-model benchmark suite across the stories15M, stories42M, and stories110M checkpoints.

Built and profiled on an **NVIDIA RTX 4060 Laptop GPU (Ada Lovelace, sm_89, 24 SMs)** as part of a GPU performance engineering research internship at IIT Jodhpur.

---

## Results at a Glance

| Model | Parameters | tok/s | GOPS | P_delta | GOPS/W (δ) | Avg kernel ms/tok |
|---|---|---|---|---|---|---|
| stories15M | 15 M | **708.81** | 20.33 | 22.5 W | 0.903 | 1.191 |
| stories42M | 42 M | **494.87** | 39.92 | 31.6 W | 1.264 | 1.794 |
| stories110M | 110 M | **241.70** | 52.78 | 40.9 W | 1.292 | 3.872 |

> GOPS/W (δ) uses baseline-subtracted workload power — the correct figure to compare against FPGA PMBus measurements that also subtract idle baseline. See [Power Methodology](#power-methodology) for why this matters.

---

## Project Structure

```
llama-gpu/
├── src/
│   ├── llama_cuda.cu        # Main CUDA C++ inference engine
│   └── analyze_profile.py   # Bottleneck analysis from CSV output
├── models/
│   ├── stories15M.bin       # LLaMA-2 checkpoint (download separately)
│   ├── stories42M.bin
│   ├── stories110M.bin
│   └── tokenizer.bin
├── results/
│   ├── llama_profile.csv    # Per-token profiling output
│   └── roofline_full.ncu-rep # Nsight Compute report
├── Kernel_reports/
│   ├── prefill/             # NCU kernel reports — prefill stage
│   └── decode/              # NCU kernel reports — decode stage
├── Makefile
└── run                      # Compiled binary
```

---

## Hardware & Software

| Item | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Architecture | Ada Lovelace (sm_89) |
| SM Count | 24 Streaming Multiprocessors |
| CUDA Cores | 3,072 FP32 (128 / SM) |
| L2 Cache | 24 MB |
| Memory | 8 GB GDDR6 · 128-bit · ~192 GB/s peak |
| CUDA Toolkit | 12.4 |
| cuBLAS | Bundled with CUDA 12.4 |
| NVML | Bundled with CUDA / nvidia-ml package |
| OS | Ubuntu 24.04 |
| Host CPU | AMD Ryzen 7 7840HS |

---

## Build

```bash
# Default: optimized build for RTX 4060 Laptop (sm_89)
make

# With Nsight Compute line-info (for ncu profiling, no runtime penalty)
make profile_build

# Full device-debug build (very slow — use with cuda-gdb only)
make debug_build

# Kaggle P100 target (sm_60)
make p100

# Check GPU info
make info
```

**Key compiler flags:**

| Flag | Reason |
|---|---|
| `-arch=sm_89` | Ada Lovelace compute capability |
| `-std=c++14` | Required for `std::thread` / `std::atomic` (PowerSampler) |
| `-lnvidia-ml` | NVML GPU power sampling |
| `-Xcompiler -pthread` | `std::thread` requires pthreads on Linux |
| `-maxrregcount=64` | Caps registers/thread → more warps resident across 24 SMs |
| `--use_fast_math` | Safe for ML inference; enables `-ftz`, `-prec-div=false` |

---

## Run

### Basic inference

```bash
./run models/stories15M.bin \
      -z models/tokenizer.bin \
      -i "Once upon a time" \
      -n 200
```

### With full profiling + power measurement

```bash
./run models/stories15M.bin \
      -z models/tokenizer.bin \
      -i "Once upon a time" \
      -n 200 \
      -r results \
      --profile
```

### All options

```
Usage:  run <checkpoint> [options]

  -t <float>   temperature [0,∞]          default 1.0
  -p <float>   top-p nucleus sampling     default 0.9
  -s <int>     random seed                default: time(NULL)
  -n <int>     steps to generate          default 256
  -i <string>  input prompt
  -z <string>  tokenizer path             default: tokenizer.bin
  -r <string>  results directory          default: results/
  --profile    enable per-token profiling + NVML power sampling
```

---

## Profiling Output

When `--profile` is passed, two reporting blocks are printed.

### Per-token kernel stats (every 8–32 tokens)

```
LLaMA-2 GPU [early] | pos= 16 | Kernel=  1.123 ms | BW= 54.08 GB/s | tok/s= 708.8
  Matmul/layer= 0.086 ms | Attn/layer= 0.009 ms | RMSNorm= 0.010 ms | RoPE= 0.004 ms | SwiGLU= 0.004 ms
  Avg kernel (post-warmup): 1.1912 ms/tok
```

Reporting cadence:
- **Early phase** (pos 9–64): every 8 tokens — attention cost is short and uniform here
- **Late phase** (pos 65+): every 32 tokens — shows O(n) attention growth over longer contexts

### Summary + GOPS/Power block

```
=== Summary ===
  Time to First Token    : 8 ms   (prefill: 5 prompt tokens)
  Tokens generated        : 193
  Decode wall time        : 0.261 s
  tok/s (decode)          : 708.81
  Warmup tokens (excl.)  : 8  (avg 1.3158 ms each)
  Measured tokens        : 185
  Avg kernel ms/tok      : 1.1912

=== GOPS / Power / Efficiency (sustained: 20 repeats, ...) ===
  Ops per forward()      : 0.030 GOP  (2*MACs over all GEMVs)
  GOPS                   : 20.329
  P_idle  (baseline)     : 4.790 W
  P_total (avg workload) : 27.296 W
  P_delta (workload-only): 22.506 W
  GOPS/W (vs P_total)    : 0.745
  GOPS/W (vs P_delta)    : 0.903
```

A CSV is written to `results/llama_profile.csv` with one row per generated token.

### Bottleneck analysis script

```bash
python3 src/analyze_profile.py results/llama_profile.csv
```

Produces a terminal bar chart of per-layer operation costs and flags the primary bottleneck (MatMul dominates at ~75% in all three models).

---

## Architecture

### What changed from Karpathy's C

| Operation | Karpathy (CPU) | This project (CUDA) |
|---|---|---|
| `matmul()` | Serial loop + OpenMP | `cublasSgemv` — vendor-tuned GEMV |
| `rmsnorm()` | Single-threaded sum | 1 block, shared-mem tree reduction |
| RoPE rotation | Serial loop over dim/2 | 1 thread per (q,k) pair |
| Multi-head attention | Nested loops | 1 CUDA block per head, inline softmax |
| SwiGLU | Serial loop | 1 thread per hidden-dim element |
| KV Cache | Heap arrays | Pre-allocated device buffers |

### Custom CUDA kernels

| Kernel | Strategy |
|---|---|
| `rmsnorm_kernel` | 1 block · shared-memory tree reduction for RMS · normalize+scale |
| `rope_kernel` | 1 thread per dimension pair · fully parallel cos/sin rotation |
| `multihead_attn_kernel` | 1 block per head · scores + softmax + V-weighted-sum in one pass |
| `swiglu_kernel` | Element-wise: `hb[i] = SiLU(hb[i]) × hb2[i]` |
| `residual_add_kernel` | Element-wise: `x[i] += y[i]` |

---

## Benchmark Results — Three Models

All runs: `./run models/<checkpoint>.bin -z models/tokenizer.bin -i "Once upon a time" -n 256 --profile`

### stories15M (dim=288, hidden=768, 6 layers, 6 heads)

```
tok/s (decode)     : 708.81
Avg kernel ms/tok  : 1.1912
GOPS               : 20.329
P_idle             : 4.790 W
P_total            : 27.296 W
P_delta            : 22.506 W
GOPS/W (P_total)   : 0.745
GOPS/W (P_delta)   : 0.903
```

### stories42M (dim=512, hidden=1376, 8 layers, 8 heads)

```
tok/s (decode)     : 494.87
Avg kernel ms/tok  : 1.7938
GOPS               : 39.921
P_idle             : 4.122 W
P_total            : 35.709 W
P_delta            : 31.587 W
GOPS/W (P_total)   : 1.118
GOPS/W (P_delta)   : 1.264
```

### stories110M (dim=768, hidden=2048, 12 layers, 12 heads)

```
tok/s (decode)     : 241.70
Avg kernel ms/tok  : 3.8724
GOPS               : 52.784
P_idle             : 6.333 W
P_total            : 47.197 W
P_delta            : 40.864 W
GOPS/W (P_total)   : 1.118
GOPS/W (P_delta)   : 1.292
```

---

## Bottleneck Analysis

### NCU findings (Nsight Compute, sm_89)

All 8 kernels (prefill + decode) land in the **memory-bound** region of the roofline — none reach the compute roof.

| Kernel | SM Throughput | Mem Throughput | DRAM | Bound By |
|---|---|---|---|---|
| `gemv2T` (GEMV) | 24.6% | 94.8% | 94.8% — 212 GB/s | **DRAM bandwidth** |
| `multihead_attn` | 2.8–3.0% | 2.8–3.0% | 1.5–2.4% | Launch under-utilisation |
| `rmsnorm_kernel` | 0.47% | 0.81% | 0.81% | Launch under-utilisation |
| `swiglu_kernel` | 0.23% | 1.84% | 1.84% | Launch under-utilisation |

**GEMV is the dominant bottleneck at 75.3% of decode compute time.** Each forward pass requires 7 GEMV calls per layer (Wq, Wk, Wv, Wo, W1, W2, W3) — each streams the full weight matrix from DRAM. L1 hit rate is only 11% on GEMV — weights are **not** L2-cached; they stream from DRAM.

**MHA, RMSNorm, and SwiGLU are launch-under-utilised.** Tiny grids (0.0–0.1 waves) leave 23+ of 24 SMs idle on every call. Warp occupancy gaps: RMSNorm 18.1% vs 93.75% theoretical; SwiGLU 15.3% vs 100%.

### Operation breakdown (stories15M, 185 tokens)

| Operation | Avg ms/layer | % of compute |
|---|---|---|
| MatMul (cuBLAS GEMV) | 0.1178 | 75.3% |
| RMSNorm | 0.0143 | 9.1% |
| Multi-Head Attention | 0.0124 | 7.9% |
| SwiGLU | 0.0061 | 3.9% |
| RoPE | 0.0059 | 3.8% |

---

## Power Methodology

Three power figures are reported because they mean different things:

| Figure | Definition | When to use |
|---|---|---|
| `P_idle` | Avg GPU power over ~0.5 s with no kernels queued | Baseline / leakage floor |
| `P_total` | Avg GPU power over the entire generation run | Compare against FPGA *total rail power* |
| `P_delta` | `P_total − P_idle` | Compare against FPGA *baseline-subtracted* power |

`GOPS/W` is reported on both bases. **Use `GOPS/W (P_delta)` if your FPGA PMBus measurement also subtracts idle baseline. Use `GOPS/W (P_total)` if your FPGA number is raw total rail power.** Mixing the two conventions overstates the GPU's efficiency advantage.

Power is sampled via `nvmlDeviceGetPowerUsage()` at 5 ms intervals on a background `std::thread`, sustained over 20 full generation repeats (3000–4000 forward() calls) to allow DVFS ramp and give stable averages.

---

## Debugging Journey

Three bugs that compiled and produced text but gave wrong answers or wrong measurements:

**Bug 1 — cuBLAS matrix layout:** Our weights are row-major (d, n). cuBLAS is column-major. Wrong `CUBLAS_OP_T`/`OP_N` combination silently computed garbage → output was incoherent text. Fixed by correctly mapping row-major (d,n) to column-major (n,d) and using `CUBLAS_OP_T` with `lda=n`.

**Bug 2 — RoPE / KV-cache ordering:** K vectors were written to the KV cache *before* RoPE rotation — every future attention step read un-rotated keys. Fixed by applying RoPE first, then writing rotated K,V to cache.

**Bug 3 — Async timing trap:** `forward()` launches async CUDA kernels. Without `cudaDeviceSynchronize()` before the wall-clock sample, the CPU timer measured launch overhead (~780 tok/s). True rate after sync: ~680–820 tok/s depending on model.

---

## Optimization Roadmap

| Priority | Target | Fix | Expected Impact |
|---|---|---|---|
| P1 | RMSNorm, SwiGLU | Multi-block grid (currently 1 block) | 20–50× kernel speedup |
| P2 | GEMV (all layers) | INT8 weight quantization | 1.5–2× GEMV throughput |
| P3 | Decode attention | Dedicated GEMV-style KV scan kernel | 10–30× attention speedup |
| P4 | RoPE | Fuse into Wq/Wk GEMV epilogue | Eliminate kernel launch |
| P5 | RMSNorm + projection | Single fused kernel | Eliminate 1 global mem RTT |
| P6 | GEMV | TF32 / FP16 Tensor Cores | Up to 8× throughput |
| P7 | All | CUDA Graphs | Reduce per-token launch overhead |

---

## Getting the Model Checkpoints

The `.bin` checkpoints are Karpathy's TinyStories models, re-exported in his binary format.

```bash
# stories15M (~60 MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -P models/

# stories42M (~170 MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin -P models/

# stories110M (~440 MB)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin -P models/

# tokenizer
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -P models/
```

---

## Citation / Reference

This work is based on:
- [Karpathy's llama2.c](https://github.com/karpathy/llama2.c) — original C inference engine
- [LLaMA 2](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023

---

## License

Code in `src/` is released under MIT. Model weights are subject to Meta's LLaMA 2 Community License.
