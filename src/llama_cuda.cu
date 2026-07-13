/*
 * llama_cuda.cu  —  LLaMA-2 inference in CUDA C++
 * Based on Karpathy's llama2.c, GPU-accelerated with bottleneck profiling.
 *
 * Metrics captured per forward pass (mirrors prefix_sum benchmark style):
 *   Kernel time | H->D | D->H | Total | BW (GB/s) | tok/s | layer breakdown
 *
 * Metrics captured per --profile run (new):
 *   GOPS | GPU power (idle / total / delta) | GOPS/W (total & delta basis)
 *   See the "GPU power sampling via NVML" comment block below for exactly
 *   what each power number means — this matters for FPGA-vs-GPU efficiency
 *   comparisons, so don't just eyeball the watts without reading it.
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -lcublas -lnvidia-ml -Xcompiler -pthread -o run src/llama_cuda.cu
 *   (change sm_75 to your GPU arch: sm_86 for RTX 30xx, sm_89 for RTX 40xx)
 *
 *   -lnvidia-ml    : NVML, used for GPU power sampling
 *   -Xcompiler -pthread : the power sampler runs on a background std::thread
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include <thread>
#include <atomic>
#include <vector>
#include <mutex>

// ─── Error-checking macros ────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[cuBLAS ERROR] %s:%d  status=%d\n",               \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define NVML_CHECK(call)                                                        \
    do {                                                                        \
        nvmlReturn_t st = (call);                                               \
        if (st != NVML_SUCCESS) {                                               \
            fprintf(stderr, "[NVML ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, nvmlErrorString(st));                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Profiling helpers ────────────────────────────────────────────────────────

static cudaEvent_t _ev_start, _ev_stop;

static inline void prof_init() {
    CUDA_CHECK(cudaEventCreate(&_ev_start));
    CUDA_CHECK(cudaEventCreate(&_ev_stop));
}
static inline void prof_destroy() {
    cudaEventDestroy(_ev_start);
    cudaEventDestroy(_ev_stop);
}
static inline void prof_start() {
    CUDA_CHECK(cudaEventRecord(_ev_start));
}
static inline float prof_stop_ms() {          // returns elapsed ms
    CUDA_CHECK(cudaEventRecord(_ev_stop));
    CUDA_CHECK(cudaEventSynchronize(_ev_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, _ev_start, _ev_stop));
    return ms;
}

// Per-forward-pass profiling counters
struct PassProfile {
    float kernel_ms;      // pure GPU compute
    float htod_ms;        // host→device transfers
    float dtoh_ms;        // device→host transfers
    float total_ms;       // wall-clock for full forward()
    float bw_GBs;         // effective memory bandwidth
    size_t bytes_moved;   // total bytes transferred (compute)

    // Per-operation breakdown (averages over all layers)
    float matmul_ms;
    float rmsnorm_ms;
    float rope_ms;
    float attn_ms;
    float swiglu_ms;
    float softmax_ms;

    int   passes;         // warmup + timed passes
};

// ─── GPU power sampling via NVML ──────────────────────────────────────────────

struct PowerSampler {
    std::thread                thr;
    std::atomic<bool>          running{false};
    std::mutex                 mtx;
    std::vector<unsigned int>  samples_mW;      // raw NVML samples, milliwatts
    nvmlDevice_t                dev;
    int                         poll_interval_us = 5000; // 5 ms between samples

    void init(int device_index) {
        NVML_CHECK(nvmlInit());
        NVML_CHECK(nvmlDeviceGetHandleByIndex(device_index, &dev));
    }

    void shutdown() {
        nvmlShutdown();
    }

    // Blocking: samples power for `ms` milliseconds and returns avg watts.
    // Used for the P_idle baseline (call with NO kernels in flight).
    float sample_idle_avg_watts(int ms) {
        int n = ms * 1000 / poll_interval_us;
        if (n < 1) n = 1;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            unsigned int mW = 0;
            nvmlDeviceGetPowerUsage(dev, &mW);
            sum += mW;
            usleep(poll_interval_us);
        }
        return (float)(sum / n) / 1000.f; // mW -> W
    }

    // Starts a background thread that polls power every poll_interval_us
    // until stop() is called. Used to cover the whole generation window
    // (P_total).
    void start() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            samples_mW.clear();
        }
        running = true;
        thr = std::thread([this]() {
            while (running.load(std::memory_order_relaxed)) {
                unsigned int mW = 0;
                nvmlDeviceGetPowerUsage(dev, &mW);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    samples_mW.push_back(mW);
                }
                usleep(poll_interval_us);
            }
        });
    }

    // Stops sampling, joins the thread, and reports avg/min/max watts plus
    // the sample count actually collected.
    void stop(float* avg_w, float* min_w, float* max_w, size_t* n) {
        running = false;
        if (thr.joinable()) thr.join();
        std::lock_guard<std::mutex> lock(mtx);
        *n = samples_mW.size();
        if (*n == 0) { *avg_w = *min_w = *max_w = 0.f; return; }
        double sum = 0;
        unsigned int mn = samples_mW[0], mx = samples_mW[0];
        for (auto v : samples_mW) {
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        *avg_w = (float)(sum / *n) / 1000.f;
        *min_w = mn / 1000.f;
        *max_w = mx / 1000.f;
    }
};

static PowerSampler g_power;

// ─── Model structs (mirrors Karpathy) ─────────────────────────────────────────

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

// Host copy of weights (memory-mapped)
typedef struct {
    float* token_embedding_table; // (vocab_size, dim)
    float* rms_att_weight;        // (layer, dim)
    float* rms_ffn_weight;        // (layer, dim)
    float* wq;                    // (layer, dim, n_heads*head_size)
    float* wk;                    // (layer, dim, n_kv_heads*head_size)
    float* wv;                    // (layer, dim, n_kv_heads*head_size)
    float* wo;                    // (layer, n_heads*head_size, dim)
    float* w1;                    // (layer, hidden_dim, dim)
    float* w2;                    // (layer, dim, hidden_dim)
    float* w3;                    // (layer, hidden_dim, dim)
    float* rms_final_weight;      // (dim,)
    float* wcls;
} TransformerWeights;

// Device (GPU) copy of weights
typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} DeviceWeights;

// GPU run-state buffers
typedef struct {
    float *x, *xb, *xb2, *hb, *hb2;
    float *q, *k, *v;
    float *att;
    float *logits;
    float *key_cache, *value_cache;
} RunState;

typedef struct {
    Config           config;
    TransformerWeights weights_h;  // host (mmap)
    DeviceWeights    weights_d;    // device
    RunState         state;
    int              fd;
    float*           data;
    ssize_t          file_size;
    cublasHandle_t   cublas;
} Transformer;

// ─── CUDA Kernels ─────────────────────────────────────────────────────────────

// RMSNorm: o[j] = weight[j] * x[j] / rms(x)
__global__ void rmsnorm_kernel(float* __restrict__ o,
                               const float* __restrict__ x,
                               const float* __restrict__ weight,
                               int size) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float ss = 0.f;
    for (int i = tid; i < size; i += blockDim.x)
        ss += x[i] * x[i];
    smem[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(smem[0] / size + 1e-5f);
    for (int i = tid; i < size; i += blockDim.x)
        o[i] = weight[i] * (rms_inv * x[i]);
}

// Softmax (in-place, single vector of length size)
__global__ void softmax_kernel(float* __restrict__ x, int size) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float val = -1e30f;
    for (int i = tid; i < size; i += blockDim.x)
        val = fmaxf(val, x[i]);
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    float sum = 0.f;
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.f / smem[0];
    for (int i = tid; i < size; i += blockDim.x)
        x[i] *= inv_sum;
}

// RoPE: rotate q and k in-place
__global__ void rope_kernel(float* __restrict__ q,
                            float* __restrict__ k,
                            int dim, int kv_dim, int head_size,
                            int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim / 2) return;
    int ii = i * 2;
    int head_dim = ii % head_size;
    float freq = 1.f / powf(10000.f, head_dim / (float)head_size);
    float val   = pos * freq;
    float fcr   = cosf(val);
    float fci   = sinf(val);

    float q0 = q[ii], q1 = q[ii + 1];
    q[ii]     = q0 * fcr - q1 * fci;
    q[ii + 1] = q0 * fci + q1 * fcr;

    if (ii < kv_dim) {
        float k0 = k[ii], k1 = k[ii + 1];
        k[ii]     = k0 * fcr - k1 * fci;
        k[ii + 1] = k0 * fci + k1 * fcr;
    }
}

// Multi-head attention: compute scores, softmax, weighted sum
// One block per head
__global__ void multihead_attn_kernel(
    float* __restrict__ xb,
    const float* __restrict__ q,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    float* __restrict__ att,
    int head_size, int kv_dim, int kv_mul,
    int seq_len, int pos,
    int loff
) {
    int h = blockIdx.x;
    extern __shared__ float smem2[];

    const float* qh  = q   + h * head_size;
    float* att_h = att + h * seq_len;
    float        scale  = rsqrtf((float)head_size);

    // 1. Dot-product scores
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        if (t <= pos) {
            const float* kh = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.f;
            for (int i = 0; i < head_size; i++)
                score += qh[i] * kh[i];
            att_h[t] = score * scale;
        } else {
            att_h[t] = -1e30f;
        }
    }
    __syncthreads();

    // 2. Softmax tree reduction
    float mx = -1e30f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x)
        mx = fmaxf(mx, att_h[t]);
    smem2[threadIdx.x] = mx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem2[threadIdx.x] = fmaxf(smem2[threadIdx.x], smem2[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smem2[0]; __syncthreads();

    float sm_sum = 0.f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        att_h[t] = expf(att_h[t] - mx);
        sm_sum  += att_h[t];
    }
    smem2[threadIdx.x] = (threadIdx.x <= pos) ? sm_sum : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem2[threadIdx.x] += smem2[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.f / (smem2[0] + 1e-6f); __syncthreads();

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        if (t <= pos) att_h[t] *= inv_sum;
        else          att_h[t] = 0.f;
    }
    __syncthreads();

    // 3. Weighted sum of values
    float* xbh = xb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t <= pos; t++) {
            const float* vh = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            acc += att_h[t] * vh[i];
        }
        xbh[i] = acc;
    }
}

// Residual add
__global__ void residual_add_kernel(float* __restrict__ x,
                                    const float* __restrict__ y,
                                    int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] += y[i];
}

// SwiGLU
__global__ void swiglu_kernel(float* __restrict__ hb,
                              const float* __restrict__ hb2,
                              int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = hb[i];
        v *= 1.f / (1.f + expf(-v));
        hb[i] = v * hb2[i];
    }
}

// ─── Helper: matmul via cuBLAS (xout = W @ x) ─────────────────────────────────
static void cublas_matmul(cublasHandle_t cublas,
                          float* xout, const float* x,
                          const float* W, int n, int d) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_T,
                             n, d,
                             &alpha,
                             W, n,
                             x, 1,
                             &beta,
                             xout, 1));
}

// ─── Op counting for GOPS ──────────────────────────────────────────────────────

static double compute_ops_per_forward(const Config* p) {
    double dim    = p->dim;
    double hidden = p->hidden_dim;
    double vocab  = p->vocab_size;
    double kv_dim = (double)(p->dim * p->n_kv_heads) / p->n_heads;

    double ops = 0;
    ops += (double)p->n_layers * (
             2.0 * dim * dim          // wq
           + 2.0 * dim * kv_dim       // wk
           + 2.0 * dim * kv_dim       // wv
           + 2.0 * dim * dim          // wo
           + 2.0 * dim * hidden       // w1
           + 2.0 * hidden * dim       // w2
           + 2.0 * dim * hidden       // w3
    );
    ops += 2.0 * dim * vocab;         // classifier head (wcls)
    return ops; // total ops for ONE token's forward() call
}

// ─── Device weight allocation ─────────────────────────────────────────────────

static void gpu_alloc_copy(float** dst, const float* src, size_t n) {
    CUDA_CHECK(cudaMalloc(dst, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
}

static void alloc_device_weights(DeviceWeights* dw, const TransformerWeights* hw, const Config* p) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;

    #define alloc_copy gpu_alloc_copy

    alloc_copy(&dw->token_embedding_table, hw->token_embedding_table, (size_t)p->vocab_size * p->dim);
    alloc_copy(&dw->rms_att_weight,        hw->rms_att_weight,        nl * p->dim);
    alloc_copy(&dw->rms_ffn_weight,        hw->rms_ffn_weight,        nl * p->dim);
    alloc_copy(&dw->wq,  hw->wq,  nl * p->dim * (p->n_heads    * head_size));
    alloc_copy(&dw->wk,  hw->wk,  nl * p->dim * (p->n_kv_heads * head_size));
    alloc_copy(&dw->wv,  hw->wv,  nl * p->dim * (p->n_kv_heads * head_size));
    alloc_copy(&dw->wo,  hw->wo,  nl * (p->n_heads * head_size) * p->dim);
    alloc_copy(&dw->w1,  hw->w1,  nl * p->dim * p->hidden_dim);
    alloc_copy(&dw->w2,  hw->w2,  nl * p->hidden_dim * p->dim);
    alloc_copy(&dw->w3,  hw->w3,  nl * p->dim * p->hidden_dim);
    alloc_copy(&dw->rms_final_weight, hw->rms_final_weight, p->dim);

    if (hw->wcls == hw->token_embedding_table)
        dw->wcls = dw->token_embedding_table;
    else
        alloc_copy(&dw->wcls, hw->wcls, (size_t)p->vocab_size * p->dim);

    #undef alloc_copy
}

static void free_device_weights(DeviceWeights* dw, int shared_cls) {
    cudaFree(dw->token_embedding_table);
    cudaFree(dw->rms_att_weight);
    cudaFree(dw->rms_ffn_weight);
    cudaFree(dw->wq); cudaFree(dw->wk); cudaFree(dw->wv); cudaFree(dw->wo);
    cudaFree(dw->w1); cudaFree(dw->w2); cudaFree(dw->w3);
    cudaFree(dw->rms_final_weight);
    if (!shared_cls) cudaFree(dw->wcls);
}

static void alloc_run_state(RunState* s, const Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUDA_CHECK(cudaMalloc(&s->x,           p->dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->xb,          p->dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->xb2,         p->dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->hb,          p->hidden_dim  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->hb2,         p->hidden_dim  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->q,           p->dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->k,           kv_dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->v,           kv_dim         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->att,         (size_t)p->n_heads * p->seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->logits,      p->vocab_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->key_cache,   (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->value_cache, (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->key_cache,   0, (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->value_cache, 0, (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
}

static void free_run_state(RunState* s) {
    cudaFree(s->x); cudaFree(s->xb); cudaFree(s->xb2);
    cudaFree(s->hb); cudaFree(s->hb2);
    cudaFree(s->q); cudaFree(s->k); cudaFree(s->v);
    cudaFree(s->att); cudaFree(s->logits);
    cudaFree(s->key_cache); cudaFree(s->value_cache);
}

// ─── Checkpoint loading (identical to Karpathy) ──────────────────────────────

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;
    w->token_embedding_table = ptr; ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;        ptr += nl * p->dim;
    w->wq = ptr;                    ptr += nl * p->dim * (p->n_heads    * head_size);
    w->wk = ptr;                    ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                    ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                    ptr += nl * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;        ptr += nl * p->dim;
    w->w1 = ptr;                    ptr += nl * p->dim * p->hidden_dim;
    w->w2 = ptr;                    ptr += nl * p->hidden_dim * p->dim;
    w->w3 = ptr;                    ptr += nl * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;      ptr += p->dim;
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void read_checkpoint(const char* checkpoint, Config* config,
                             TransformerWeights* weights,
                             int* fd, float** data, ssize_t* file_size) {
    FILE *f = fopen(checkpoint, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", checkpoint); exit(1); }
    if (fread(config, sizeof(Config), 1, f) != 1) { exit(1); }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(f, 0, SEEK_END); *file_size = ftell(f); fclose(f);
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed\n"); exit(1); }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    memory_map_weights(weights, config, *data + sizeof(Config)/sizeof(float), shared_weights);
}

static void build_transformer(Transformer* t, const char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights_h,
                    &t->fd, &t->data, &t->file_size);

    float hd_ms_start = 0;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    alloc_device_weights(&t->weights_d, &t->weights_h, &t->config);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&hd_ms_start, ev0, ev1));
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    fprintf(stderr, "[init] Weights uploaded to GPU in %.3f ms\n", hd_ms_start);

    alloc_run_state(&t->state, &t->config);
    CUBLAS_CHECK(cublasCreate(&t->cublas));
}

static void free_transformer(Transformer* t) {
    int shared = (t->weights_h.wcls == t->weights_h.token_embedding_table);
    free_device_weights(&t->weights_d, shared);
    free_run_state(&t->state);
    cublasDestroy(t->cublas);
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
}

// ─── GPU Forward pass with per-op profiling ───────────────────────────────────

static float* forward(Transformer* transformer, int token, int pos,
                      PassProfile* prof, int do_profile) {
    Config* p    = &transformer->config;
    DeviceWeights* w = &transformer->weights_d;
    RunState*   s    = &transformer->state;

    int dim        = p->dim;
    int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul     = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size  = dim / p->n_heads;

    int tb_dim    = (dim    < 1024) ? dim    : 1024;
    int tb_hidden = (hidden_dim < 1024) ? hidden_dim : 1024;

    cudaEvent_t ev_total_start, ev_total_stop;
    if (do_profile) {
        CUDA_CHECK(cudaEventCreate(&ev_total_start));
        CUDA_CHECK(cudaEventCreate(&ev_total_stop));
        CUDA_CHECK(cudaEventRecord(ev_total_start));
    }

    CUDA_CHECK(cudaMemcpy(s->x,
                          w->token_embedding_table + token * dim,
                          dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    float matmul_accum = 0, rmsnorm_accum = 0, rope_accum = 0;
    float attn_accum = 0, swiglu_accum = 0;

    for (int l = 0; l < p->n_layers; l++) {
        int loff = l * p->seq_len * kv_dim;

        // ── Attention RMSNorm ─────────────────────────────────────────────
        if (do_profile) prof_start();
        rmsnorm_kernel<<<1, tb_dim, tb_dim * sizeof(float)>>>(
            s->xb, s->x, w->rms_att_weight + l * dim, dim);
        if (do_profile) rmsnorm_accum += prof_stop_ms();

        // ── QKV matmuls ───────────────────────────────────────────────────
        if (do_profile) prof_start();
        cublas_matmul(transformer->cublas, s->q,  s->xb, w->wq + l*dim*dim,         dim, dim);
        cublas_matmul(transformer->cublas, s->k,  s->xb, w->wk + l*dim*kv_dim,      dim, kv_dim);
        cublas_matmul(transformer->cublas, s->v,  s->xb, w->wv + l*dim*kv_dim,      dim, kv_dim);
        if (do_profile) matmul_accum += prof_stop_ms();

        // ── RoPE (before KV cache write — rotated K must be cached) ───────
        {
            int nthreads = 256;
            int nblocks  = (dim/2 + nthreads - 1) / nthreads;
            if (do_profile) prof_start();
            rope_kernel<<<nblocks, nthreads>>>(s->q, s->k, dim, kv_dim, head_size, pos);
            if (do_profile) rope_accum += prof_stop_ms();
        }

        // ── Write RoPE-rotated K and V into KV cache ──────────────────────
        CUDA_CHECK(cudaMemcpy(s->key_cache   + loff + pos * kv_dim,
                              s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s->value_cache + loff + pos * kv_dim,
                              s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        // ── Multi-head attention ──────────────────────────────────────────
        {
            int nthreads_attn = 256;

            if (do_profile) prof_start();
            multihead_attn_kernel<<<p->n_heads, nthreads_attn, nthreads_attn * sizeof(float)>>>(
                s->xb, s->q, s->key_cache, s->value_cache, s->att,
                head_size, kv_dim, kv_mul, p->seq_len, pos, loff);
            if (do_profile) attn_accum += prof_stop_ms();
        }

        // ── Attention output projection ───────────────────────────────────
        if (do_profile) prof_start();
        cublas_matmul(transformer->cublas, s->xb2, s->xb,
                      w->wo + l * dim * dim, dim, dim);
        if (do_profile) matmul_accum += prof_stop_ms();

        // ── Residual add ──────────────────────────────────────────────────
        residual_add_kernel<<<(dim+255)/256, 256>>>(s->x, s->xb2, dim);

        // ── FFN RMSNorm ───────────────────────────────────────────────────
        if (do_profile) prof_start();
        rmsnorm_kernel<<<1, tb_dim, tb_dim * sizeof(float)>>>(
            s->xb, s->x, w->rms_ffn_weight + l * dim, dim);
        if (do_profile) rmsnorm_accum += prof_stop_ms();

        // ── FFN matmuls (w1, w3) ──────────────────────────────────────────
        if (do_profile) prof_start();
        cublas_matmul(transformer->cublas, s->hb,  s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        cublas_matmul(transformer->cublas, s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        if (do_profile) matmul_accum += prof_stop_ms();

        // ── SwiGLU ────────────────────────────────────────────────────────
        if (do_profile) prof_start();
        swiglu_kernel<<<(hidden_dim+255)/256, 256>>>(s->hb, s->hb2, hidden_dim);
        if (do_profile) swiglu_accum += prof_stop_ms();

        // ── FFN w2 ────────────────────────────────────────────────────────
        if (do_profile) prof_start();
        cublas_matmul(transformer->cublas, s->xb, s->hb,
                      w->w2 + l * hidden_dim * dim, hidden_dim, dim);
        if (do_profile) matmul_accum += prof_stop_ms();

        // ── Residual add ──────────────────────────────────────────────────
        residual_add_kernel<<<(dim+255)/256, 256>>>(s->x, s->xb, dim);
    }

    // ── Final RMSNorm + classifier ─────────────────────────────────────────
    if (do_profile) prof_start();
    rmsnorm_kernel<<<1, tb_dim, tb_dim * sizeof(float)>>>(
        s->x, s->x, w->rms_final_weight, dim);
    if (do_profile) rmsnorm_accum += prof_stop_ms();

    if (do_profile) prof_start();
    cublas_matmul(transformer->cublas, s->logits, s->x, w->wcls, dim, p->vocab_size);
    if (do_profile) matmul_accum += prof_stop_ms();

    CUDA_CHECK(cudaDeviceSynchronize());

    if (do_profile) {
        CUDA_CHECK(cudaEventRecord(ev_total_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_total_stop));
        float total_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_stop));

        prof->kernel_ms   = total_ms;
        prof->matmul_ms   = matmul_accum  / p->n_layers;
        prof->rmsnorm_ms  = rmsnorm_accum / p->n_layers;
        prof->rope_ms     = rope_accum    / p->n_layers;
        prof->attn_ms     = attn_accum    / p->n_layers;
        prof->swiglu_ms   = swiglu_accum  / p->n_layers;

        size_t nl = p->n_layers;
        size_t head_sz = p->dim / p->n_heads;
        size_t weight_bytes =
            nl * ((size_t)p->dim * p->n_heads    * head_sz * 2 +
                  (size_t)p->dim * p->n_kv_heads * head_sz * 2 +
                  (size_t)p->dim * p->hidden_dim * 3) * sizeof(float);
        weight_bytes += (size_t)p->dim * p->vocab_size * sizeof(float);
        prof->bytes_moved = weight_bytes;
        prof->bw_GBs      = (weight_bytes / 1e9f) / (total_ms / 1e3f);

        cudaEventDestroy(ev_total_start);
        cudaEventDestroy(ev_total_stop);
    }

    return s->logits;
}

// ─── Tokenizer (identical to Karpathy) ────────────────────────────────────────

typedef struct { char *str; int id; } TokenIndex;
typedef struct {
    char** vocab; float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size; unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

static void build_tokenizer(Tokenizer* t, const char* path, int vocab_size) {
    t->vocab_size   = vocab_size;
    t->vocab        = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i*2]   = (unsigned char)i;
        t->byte_pieces[i*2+1] = '\0';
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open tokenizer %s\n", path); exit(1); }
    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) exit(1);
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, f) != 1) exit(1);
        int len; if (fread(&len, sizeof(int), 1, f) != 1) exit(1);
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, f) != 1) exit(1);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

static void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

static void safe_printf(char *piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char b = piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sv, int vs) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sv, vs, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

static void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos,
                   int *tokens, int *n_tokens) {
    if (!text) { fprintf(stderr, "NULL text\n"); exit(1); }
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* buf = (char*)malloc((t->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') tokens[(*n_tokens)++] = str_lookup(" ", t->sorted_vocab, t->vocab_size);

    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        buf[str_len++] = *c; buf[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else for (int i = 0; i < (int)str_len; i++)
            tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3;
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10; int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            sprintf(buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < *n_tokens-1; i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}

// ─── Sampler (identical to Karpathy) ─────────────────────────────────────────

typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size; ProbIndex* probindex;
    float temperature, topp; unsigned long long rng_state;
} Sampler;

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.f;
}
static int sample_argmax(float* p, int n) {
    int mi = 0; for (int i = 1; i < n; i++) if (p[i] > p[mi]) mi = i; return mi;
}
static int sample_mult(float* p, int n, float coin) {
    float cdf = 0; for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}
static int cmp_prob(const void* a, const void* b) {
    float d = ((ProbIndex*)a)->prob - ((ProbIndex*)b)->prob;
    return (d > 0) ? -1 : (d < 0) ? 1 : 0;
}
static int sample_topp(float* probs, int n, float topp, ProbIndex* pi, float coin) {
    int n0 = 0; float cutoff = (1.f - topp) / (n - 1);
    for (int i = 0; i < n; i++) if (probs[i] >= cutoff) { pi[n0].index = i; pi[n0].prob = probs[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), cmp_prob);
    float cum = 0; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += pi[i].prob; if (cum > topp) { last = i; break; } }
    float r = coin * cum, cdf = 0;
    for (int i = 0; i <= last; i++) { cdf += pi[i].prob; if (r < cdf) return pi[i].index; }
    return pi[last].index;
}

static void build_sampler(Sampler* s, int vs, float temp, float topp, unsigned long long seed) {
    s->vocab_size = vs; s->temperature = temp; s->topp = topp; s->rng_state = seed;
    s->probindex = (ProbIndex*)malloc(vs * sizeof(ProbIndex));
}
static void free_sampler(Sampler* s) { free(s->probindex); }

static int sample(Sampler* s, float* d_logits, int vocab_size) {
    float* h_logits = (float*)malloc(vocab_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    int next;
    if (s->temperature == 0.f) {
        next = sample_argmax(h_logits, vocab_size);
    } else {
        for (int i = 0; i < vocab_size; i++) h_logits[i] /= s->temperature;
        float mx = h_logits[0];
        for (int i = 1; i < vocab_size; i++) if (h_logits[i] > mx) mx = h_logits[i];
        float sum = 0;
        for (int i = 0; i < vocab_size; i++) { h_logits[i] = expf(h_logits[i] - mx); sum += h_logits[i]; }
        for (int i = 0; i < vocab_size; i++) h_logits[i] /= sum;
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0 || s->topp >= 1)
            next = sample_mult(h_logits, vocab_size, coin);
        else
            next = sample_topp(h_logits, vocab_size, s->topp, s->probindex, coin);
    }
    free(h_logits);
    return next;
}

// ─── Timing util ─────────────────────────────────────────────────────────────

static long time_ms() {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

// ─── Generate loop with per-token profiling + TTFT ─────────────────────────────

// Returns the number of forward() calls made in this generate() run, so
// main() can sum them across --repeat calls for the combined GOPS figure.
int generate(Transformer* t, Tokenizer* tok, Sampler* sam,
             const char* prompt, int steps,
             int profile, const char* results_dir) {
    const char* empty = "";
    if (!prompt) prompt = empty;

    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    int  num_prompt_tokens = 0;
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "encode error\n"); exit(1); }

    // Warm-up: one pass (not counted) — primes cuBLAS workspace / L2 cache.
    // This is run BEFORE the TTFT clock starts, so warmup cost never
    // pollutes the TTFT measurement.
    {
        PassProfile dummy = {};
        forward(t, prompt_tokens[0], 0, &dummy, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // NOTE: the P_idle baseline sample and the continuous power-sampling
    // thread now live in main(), wrapped around however many back-to-back
    // generate() calls --repeat asks for. That way the power window spans
    // the whole sustained-load run instead of resetting on every call —
    // see print_gops_power_summary() right after this function.

    FILE* csv = NULL;
    if (profile && results_dir) {
        char path[512]; snprintf(path, sizeof(path), "%s/llama_profile.csv", results_dir);
        csv = fopen(path, "w");
        if (csv) {
            fprintf(csv,
                "pos,token_id,kernel_ms,bw_GBs,"
                "matmul_ms_per_layer,rmsnorm_ms_per_layer,rope_ms_per_layer,"
                "attn_ms_per_layer,swiglu_ms_per_layer\n");
        }
    }

    printf("\n=== LLaMA-2 CUDA Inference — Token Generation ===\n\n");

    // ── TTFT (Time To First Token) ───────────────────────────────────────────

    long ttft_start = time_ms();
    long ttft_ms    = -1;   // -1 = not yet recorded

    long   start  = time_ms();   // decode-phase wall-clock start
    int    next, token = prompt_tokens[0];
    int    pos = 0;
    double total_kernel_ms    = 0;
    int    counted_tokens     = 0;
    double warmup_kernel_ms   = 0;

    const int WARMUP_TOKENS  = 8;
    const int EARLY_INTERVAL = 8;
    const int LATE_INTERVAL  = 32;
    const int LATE_THRESHOLD = 64;

    // total_forward_calls counts forward() calls made in THIS generate()
    // call only. When --repeat > 1, main() sums this return value across
    // every repeat to get the true total for the combined GOPS figure.
    int  total_forward_calls = 0;

    while (pos < steps) {
        PassProfile prof = {};
        float* logits = forward(t, token, pos, &prof, profile);
        total_forward_calls++;


        CUDA_CHECK(cudaDeviceSynchronize());

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sam, logits, t->config.vocab_size);
        }
        pos++;

        if (next == 1) break;

        char* piece = decode(tok, token, next);
        safe_printf(piece); fflush(stdout);
        token = next;

        // Record TTFT the first time pos reaches num_prompt_tokens — i.e.
        // prefill (forced prompt tokens) is complete and the first
        // GENERATED token has just been produced.
        if (ttft_ms < 0 && pos >= num_prompt_tokens) {
            ttft_ms = time_ms() - ttft_start;
            // Reset decode-phase timer so prefill cost doesn't dilute
            // steady-state decode tok/s.
            start = time_ms();
        }

        if (profile) {
            if (csv)
                fprintf(csv, "%d,%d,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        pos, token,
                        prof.kernel_ms, prof.bw_GBs,
                        prof.matmul_ms, prof.rmsnorm_ms, prof.rope_ms,
                        prof.attn_ms,   prof.swiglu_ms);

            if (pos <= WARMUP_TOKENS) {
                warmup_kernel_ms += prof.kernel_ms;
            } else {
                total_kernel_ms += prof.kernel_ms;
                counted_tokens++;
            }

            int is_last     = (pos == steps - 1) || (next == 1);
            int print_early = (pos > WARMUP_TOKENS) &&
                              (pos <= LATE_THRESHOLD) &&
                              (pos % EARLY_INTERVAL == 0);
            int print_late  = (pos > LATE_THRESHOLD) &&
                              (pos % LATE_INTERVAL == 0);

            if (print_early || print_late || is_last) {
                long now = time_ms();
                double elapsed_s = (now - start) / 1000.0;
                double toks_s    = (counted_tokens > 0 && elapsed_s > 0)
                                 ? counted_tokens / elapsed_s : 0;
                double avg_ms    = (counted_tokens > 0)
                                 ? total_kernel_ms / counted_tokens : 0;

                const char* phase = (pos <= LATE_THRESHOLD) ? "early" : "late ";
                fprintf(stderr,
                    "\nLLaMA-2 GPU [%s] | pos=%3d | "
                    "Kernel= %6.3f ms | BW= %5.2f GB/s | tok/s= %.1f\n"
                    "  Matmul/layer= %.3f ms | Attn/layer= %.3f ms | "
                    "RMSNorm= %.3f ms | RoPE= %.3f ms | SwiGLU= %.3f ms\n"
                    "  Avg kernel (post-warmup): %.4f ms/tok\n",
                    phase, pos,
                    prof.kernel_ms, prof.bw_GBs, toks_s,
                    prof.matmul_ms, prof.attn_ms,
                    prof.rmsnorm_ms, prof.rope_ms, prof.swiglu_ms,
                    avg_ms);
            }
        }
    }

    printf("\n");

    if (pos > 1) {
        long end  = time_ms();
        double elapsed_s = (end - start) / 1000.0;
        double tok_s;
        if (elapsed_s > 0) {
            tok_s = (counted_tokens > 0) ? counted_tokens / elapsed_s
                                          : (pos - 1) / elapsed_s;
        } else {
            tok_s = 0;
        }

        fprintf(stderr,
            "\n=== Summary ===\n"
            "  Time to First Token    : %ld ms   (prefill: %d prompt token%s)\n"
            "  Tokens generated        : %d\n"
            "  Decode wall time        : %.3f s\n"
            "  tok/s (decode)          : %.2f\n",
            ttft_ms, num_prompt_tokens, (num_prompt_tokens == 1 ? "" : "s"),
            pos - 1,
            elapsed_s,
            tok_s);

        if (profile) {
            fprintf(stderr,
                "  Warmup tokens (excl.)  : %d  (avg %.4f ms each)\n"
                "  Measured tokens        : %d\n"
                "  Avg kernel ms/tok      : %.4f\n",
                WARMUP_TOKENS,
                (WARMUP_TOKENS > 0) ? warmup_kernel_ms / WARMUP_TOKENS : 0.0,
                counted_tokens,
                (counted_tokens > 0) ? total_kernel_ms / counted_tokens : 0.0);
        }
    }

    if (csv) {
        fclose(csv);
        fprintf(stderr, "  Profile CSV            : %s/llama_profile.csv\n", results_dir);
    }

    free(prompt_tokens);
    return total_forward_calls;
}

// ─── Combined GOPS / power summary across N back-to-back generate() calls ────

static void print_gops_power_summary(const Config* cfg, int repeats,
                                     int total_forward_calls, double total_wall_s,
                                     float idle_power_w, float total_power_w,
                                     float min_power_w, float max_power_w,
                                     size_t power_samples) {
    double ops_per_fwd = compute_ops_per_forward(cfg);
    double total_ops   = ops_per_fwd * total_forward_calls;
    double GOPS = (total_wall_s > 0) ? (total_ops / total_wall_s) / 1e9 : 0.0;

    double delta_power_w = total_power_w - idle_power_w;
    if (delta_power_w < 0) delta_power_w = 0; // guard against sampling noise

    double gops_per_w_total = (total_power_w > 0) ? GOPS / total_power_w : 0.0;
    double gops_per_w_delta = (delta_power_w  > 0) ? GOPS / delta_power_w : 0.0;

    fprintf(stderr,
        "\n=== GOPS / Power / Efficiency "
        "(sustained: %d repeat%s, %d forward() calls, %.3f s, %zu power samples) ===\n"
        "  Ops per forward()      : %.3f GOP  (2*MACs over all GEMVs, matches FPGA GOPS convention)\n"
        "  GOPS                   : %.3f\n"
        "  P_idle  (baseline)     : %.3f W   <- GPU idle, no kernels running\n"
        "  P_total (avg, workload): %.3f W   <- total board power during the whole run (P_idle is INCLUDED in this)\n"
        "  P_delta (workload-only): %.3f W   <- P_total - P_idle; power attributable to compute alone\n"
        "  P_total min/max        : %.3f / %.3f W\n"
        "  GOPS/W  (vs P_total)   : %.3f   <- use this if your FPGA number is also total board/rail power\n"
        "  GOPS/W  (vs P_delta)   : %.3f   <- use this if your FPGA number is baseline-subtracted workload power\n",
        repeats, (repeats == 1 ? "" : "s"), total_forward_calls, total_wall_s, power_samples,
        ops_per_fwd / 1e9,
        GOPS,
        idle_power_w,
        total_power_w,
        delta_power_w,
        min_power_w, max_power_w,
        gops_per_w_total,
        gops_per_w_delta);

    if (total_wall_s < 2.0) {
        fprintf(stderr,
            "  NOTE: sustained window is only %.3f s — for a trustworthy P_delta, aim\n"
            "        for several seconds (raise --repeat). Below ~1-2 s, P_delta is\n"
            "        likely dominated by power-sensor refresh lag / DVFS ramp rather\n"
            "        than real dynamic power.\n", total_wall_s);
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

static void error_usage() {
    fprintf(stderr,
        "Usage:   run <checkpoint> [options]\n"
        "Example: run models/stories15M.bin -z models/tokenizer.bin -i \"Once upon a time\"\n"
        "Options:\n"
        "  -t <float>   temperature [0,∞], default 1.0\n"
        "  -p <float>   top-p nucleus sampling [0,1], default 0.9\n"
        "  -s <int>     random seed (default: time)\n"
        "  -n <int>     steps to generate, default 256\n"
        "  -i <string>  input prompt\n"
        "  -z <string>  tokenizer path (default: tokenizer.bin)\n"
        "  -r <string>  results directory (default: results/)\n"
        "  --profile    enable per-token bottleneck profiling + GOPS/power/GOPS-W\n"
        "  --repeat <int> back-to-back generate() calls under ONE power window\n"
        "                 (default: 1). Use this to sustain load for several\n"
        "                 seconds so P_delta / GOPS-per-W-delta is trustworthy —\n"
        "                 a single short run isn't long enough for the GPU's\n"
        "                 power sensor to refresh or for clocks to ramp.\n"
    );
    exit(1);
}

int main(int argc, char* argv[]) {
    char* checkpoint_path = NULL;
    char* tokenizer_path  = (char*)"models/tokenizer.bin";
    char* results_dir     = (char*)"results";
    char* prompt          = NULL;
    float temperature     = 1.0f;
    float topp            = 0.9f;
    int   steps           = 256;
    unsigned long long rng_seed = 0;
    int   do_profile      = 0;
    int   repeats         = 1;

    if (argc < 2) error_usage();
    checkpoint_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) { do_profile = 1; continue; }
        if (i + 1 >= argc) error_usage();
        if      (strcmp(argv[i], "-t") == 0) { temperature = atof(argv[++i]); }
        else if (strcmp(argv[i], "-p") == 0) { topp        = atof(argv[++i]); }
        else if (strcmp(argv[i], "-s") == 0) { rng_seed    = atoll(argv[++i]); }
        else if (strcmp(argv[i], "-n") == 0) { steps       = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-i") == 0) { prompt      = argv[++i]; }
        else if (strcmp(argv[i], "-z") == 0) { tokenizer_path = argv[++i]; }
        else if (strcmp(argv[i], "-r") == 0) { results_dir = argv[++i]; }
        else if (strcmp(argv[i], "--repeat") == 0) { repeats = atoi(argv[++i]); }
        else error_usage();
    }

    if (rng_seed == 0) rng_seed = (unsigned long long)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || topp > 1) topp = 0.9f;
    if (steps <= 0) steps = 0;
    if (repeats <= 0) repeats = 1;

    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "\n[GPU] %s  |  %zu MB VRAM  |  %d SMs  |  CC %d.%d\n\n",
            prop.name,
            prop.totalGlobalMem / (1024*1024),
            prop.multiProcessorCount,
            prop.major, prop.minor);

    prof_init();
    if (do_profile) g_power.init(dev);

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len;

    Config* c = &transformer.config;
    fprintf(stderr,
        "[Model] dim=%d  hidden=%d  layers=%d  heads=%d  kv_heads=%d  vocab=%d  seq=%d\n\n",
        c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size, c->seq_len);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // ── P_idle baseline ──────────────────────────────────────────────────────
    // Sampled once, right here: no kernels have been launched yet in this
    // process, so the GPU is genuinely idle. This is the static/leakage
    // floor that P_total (below) will have baked into it.
    float idle_power_w = 0.f;
    if (do_profile) {
        idle_power_w = g_power.sample_idle_avg_watts(500);
        fprintf(stderr, "[power] GPU idle baseline (P_idle): %.3f W\n", idle_power_w);
        if (repeats > 1)
            fprintf(stderr, "[power] Running %d back-to-back generations under one power window...\n", repeats);
    }

    // ── Sustained-load window: repeats back-to-back generate() calls, ONE
    // continuous power-sampling window and ONE wall-clock timer spanning
    // all of them. This is what --repeat is for — a single short run
    // doesn't give the GPU's power sensor time to refresh or clocks time
    // to ramp, so P_delta comes out as noise (see print_gops_power_summary).
    int  total_forward_calls = 0;
    long bench_start = time_ms();
    if (do_profile) g_power.start();

    for (int r = 0; r < repeats; r++) {
        if (repeats > 1) fprintf(stderr, "\n[repeat %d/%d]\n", r + 1, repeats);
        total_forward_calls += generate(&transformer, &tokenizer, &sampler,
                                        prompt, steps, do_profile, results_dir);
    }

    long bench_end = time_ms();
    if (do_profile) {
        float total_power_w = 0.f, min_power_w = 0.f, max_power_w = 0.f;
        size_t power_samples = 0;
        g_power.stop(&total_power_w, &min_power_w, &max_power_w, &power_samples);

        double total_wall_s = (bench_end - bench_start) / 1000.0;
        print_gops_power_summary(&transformer.config, repeats, total_forward_calls,
                                 total_wall_s, idle_power_w, total_power_w,
                                 min_power_w, max_power_w, power_samples);
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    prof_destroy();
    if (do_profile) g_power.shutdown();
    return 0;
}