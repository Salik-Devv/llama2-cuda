# ── LLaMA-2 CUDA Makefile — RTX 4060 Laptop (Ada Lovelace, 24 SM) ───────────
#
# Changes from previous version:
#   + -lnvidia-ml        : NVML library for GPU power sampling (PowerSampler)
#   + -Xcompiler -pthread: std::thread used by the background power sampler
#   + -std=c++14         : std::thread / std::atomic / std::mutex require C++11+
#
# Usage:
#   make               → optimized build for RTX 4060 Laptop (sm_89)
#   make profile_build → adds -lineinfo for Nsight Compute (ncu)
#   make debug_build   → adds -G for full device debug (very slow)
#   make p100          → Kaggle P100 (sm_60)
#   make clean
#   make info          → nvidia-smi GPU summary

NVCC   := nvcc
TARGET := run
SRC    := src/llama_cuda.cu
ARCH   := sm_89

# -maxrregcount=64  : caps regs/thread so 24 SMs hold more warps resident.
# --use_fast_math   : -ftz, -prec-div=false, -prec-sqrt=false (safe for ML).
# -std=c++14        : required for std::thread / std::atomic / std::mutex.
# -Xcompiler -pthread : links pthreads for std::thread on Linux.
NVCCFLAGS := \
	-arch=$(ARCH) \
	-O3 \
	--use_fast_math \
	-std=c++14 \
	-maxrregcount=64 \
	-Xcompiler -O3 \
	-Xcompiler -march=native \
	-Xcompiler -pthread

# -lnvidia-ml : NVML for nvmlDeviceGetPowerUsage() in PowerSampler
LIBS := -lcublas -lnvidia-ml

.PHONY: all profile_build debug_build p100 clean info

all: $(TARGET)
	@echo ""
	@echo "  ✓  $(TARGET)  [$(ARCH), 24 SM, RTX 4060 Laptop]"
	@echo "     ./$(TARGET) models/stories15M.bin -z models/tokenizer.bin -i \"Once upon a time\" --profile"
	@echo ""

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

# Adds -lineinfo: source-line mapping for ncu/Nsight without full debug penalty
profile_build: $(SRC)
	$(NVCC) $(NVCCFLAGS) -lineinfo $< -o $(TARGET) $(LIBS)
	@echo "  ✓  profile build done — run with: ncu ./$(TARGET) ..."

# Full device debug (very slow runtime; needed for cuda-gdb)
debug_build: $(SRC)
	$(NVCC) -arch=$(ARCH) -std=c++14 -G -O0 \
	        -Xcompiler -pthread \
	        $< -o $(TARGET) $(LIBS)
	@echo "  ✓  debug build done"

# Kaggle P100 (sm_60)
p100: $(SRC)
	$(NVCC) -arch=sm_60 -O3 --use_fast_math \
	        -std=c++14 \
	        -Xcompiler -O3 \
	        -Xcompiler -pthread \
	        $< -o $(TARGET) $(LIBS)
	@echo "  ✓  built for P100 (sm_60)"

clean:
	rm -f $(TARGET)

info:
	@nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free \
	            --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not found"