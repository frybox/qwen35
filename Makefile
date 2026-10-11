# qwen35 — one command builds BOTH binaries (CPU + CUDA) into zig-out/bin/.
#
#   make          -> zig-out/bin/qwen35        (CPU,   via `zig build`)
#                   zig-out/bin/qwen35-cuda   (CUDA,  via zig cc + nvcc)
#   make test     -> `zig build test` (CPU unit tests, must stay green)
#   make run0.8b  / run4b / run0.8b-cuda / run4b-cuda  (convenience)
#
# CUDA two-step chain (mirrors /mnt/llm-dev/llama2/zig): zig cc can't emit a
# .cu and zig's lld can't resolve CUB device symbols, so nvcc compiles
# engine_cuda.cu AND does the final link of the Zig host object.
ZIG := $(shell command -v zig || echo /home/wisedu/software/zig-x86_64-linux-0.17.0/zig)
NVCC := $(shell command -v nvcc || echo /usr/local/cuda/bin/nvcc)
HOST_OBJ := .zig-cache/qwen35_cuda_host.o
CUDA_EXE := zig-out/bin/qwen35-cuda

.PHONY: all build cpu cuda test run0.8b run2b run4b run0.8b-cuda run2b-cuda run4b-cuda

all: build cpu cuda

build:
	$(ZIG) build

cpu:
	$(ZIG) build

cpufast:
	$(ZIG) build -Doptimize=ReleaseFast

cuda: $(CUDA_EXE)

$(CUDA_EXE): $(HOST_OBJ) src/kernel.cu
	$(NVCC) -O3 -arch=native -o=$@ $(HOST_OBJ) src/kernel.cu -lcudart -lm

$(HOST_OBJ): src/main_cuda.zig src/engine.zig src/tokenizer.zig
	mkdir -p $(dir $@)
	$(ZIG) cc -c $< -o $@

test:
	$(ZIG) build test

run0.8b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 中国首都在哪里？

run2b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 中国首都在哪里？

run4b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 中国首都在哪里？

run0.8b-cuda:
	CUDA_VISIBLE_DEVICES=7 $(CUDA_EXE) run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 中国首都在哪里？

run2b-cuda:
	CUDA_VISIBLE_DEVICES=7 $(CUDA_EXE) run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 中国首都在哪里？

run4b-cuda:
	CUDA_VISIBLE_DEVICES=7 $(CUDA_EXE) run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 中国首都在哪里？
