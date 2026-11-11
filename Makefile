# qwen35 — one command builds the CLI, optionally with the CUDA backend linked in.
#
#   make          -> zig-out/bin/qwen35        (CPU only, via `zig build`)
#   make cuda     -> zig-out/bin/qwen35        (same binary, CUDA kernels linked in,
#                    via `zig build -Dcuda=true`: nvcc -c -> nvcc -dlink ->
#                    addObjectFile.  There is no kernel .so -- one executable.)
#   make test     -> `zig build test` (CPU unit tests, must stay green)
#   make run0.8b / run4b / run0.8b-cuda / run4b-cuda  (convenience)
#
# There is no legacy CUDA binary any more: the retired whole-model GPU forward
# (src/main_cuda.zig + src/kernel.cu + src/kernel_gdn.cu + src/kernel_matvec.cu)
# was deleted.  `--backend cuda` on the single CLI is the only CUDA path.  See
# git history for the deleted files.
ZIG := $(shell command -v zig || echo /home/wisedu/software/zig-x86_64-linux-0.17.0/zig)
# Build cache on LOCAL disk: the project .zig-cache lives on NFS (/mnt/LLMs)
# and zig intermittently fails with "failed to write cache manifest: Unexpected"
# there. /tmp is local and fast.
ZIGCACHE := --cache-dir /tmp/qwen35-cache

# Long-prompt targets (`*-long`: the ~700-token tea essay prompt, 600+ chars out).
#
# Why an explicit -c is mandatory: new llama.cpp builds default to `-c 0`
# = "loaded from model", i.e. the model's TRAINED context (131072 for this
# 27B).  f16 KV is ~128 KB/token, so the KV cache alone asks for 16384 MiB;
# with the 10.9 GB IQ3_XXS weights that is >24 GB and cudaMalloc fails on the
# 4090D (24 GiB) with "failed to allocate buffer for kv cache".  The essay
# run needs ~1k prompt + ~1.2k generated tokens, so 4096 is plenty and drops
# the KV cache to ~512 MiB.
#
# -n is enforced here too so the budget is stated in one place; --single-turn
# keeps llama-cli out of its REPL (without it a non-interactive run can read
# nothing, print nothing and still exit 0 -- looks like the model never ran).
#
# -ctxcp 0 -ub 2048 (2026-11, measured on 27B IQ3_XXS / 4090D / -t 1):
#   llama-cli's own "prompt eval" line is NOT the prefill that llama-bench's
#   pp<N> reports, because its timing window (server-common.h t_prompt_ms =
#   t_prompt_last - t_start, set in server-context.cpp at n_gen == 1) also
#   swallows work that is not the forward pass:
#     - context checkpoints (-ctxcp, default 32) copy a 149.6 MiB KV state and
#       force the prompt into ubatch-aligned chunks -- `llama-cli -v` shows the
#       557-token prompt split as 41 / 1 / 512 / 4;
#     - common_prompt_batch_decode (common/common.cpp) decodes the last token
#       on its own when saving session state, adding the stray 1- and 4-token
#       fragments, and small matvecs are pure memory-bound on GPU;
#     - n_ubatch defaults to 512, so any prompt > 512 pays a second, tiny tail
#       batch as well.
#   Same prompt, same model, same machine:
#     llama-cli (defaults)                 848-861 t/s   <- what we used to see
#     llama-cli -ctxcp 0                  1248 t/s
#     llama-cli -ctxcp 0 -ub 2048         1389-1393 t/s
#     llama-bench -p 557 -r 5              1778 t/s      (pure forward + warmup)
#     llama-bench -p 512 -r 5              2044 t/s
#   Removing the non-compute overhead makes llama-cli comparable with
#   llama-bench (the remaining gap is bench's warmup amortising the one-time
#   sched_reserve, ~43 ms, and its r=5 averaging).  Use llama-bench for
#   hardware/kernel numbers and these flags when you want llama-cli to measure
#   prompt processing rather than checkpoint bookkeeping.
LONG_CTX  := -c 4096
LONG_NGEN := -n 1200
LONG_CLI  := -ctxcp 0 -ub 2048
LONG_ARGS := $(LONG_CTX) $(LONG_NGEN) -t 1 --single-turn $(LONG_CLI)

.PHONY: all build cpu cpufast cudabackend cuda cuda-check test run0.8b run2b run4b run27b run27bq8 run27bq4 run27biq4xs run27bq4kxl run27bq8kl run27bq8kxl run27biq3xxs run27biq2xxs run27biq1s run27b-long run0.8b-cuda run2b-cuda run4b-cuda

all: build cpu cuda

build:
	$(ZIG) build $(ZIGCACHE)

cpu:
	$(ZIG) build $(ZIGCACHE)

cpufast:
	$(ZIG) build -Doptimize=ReleaseFast $(ZIGCACHE)

# `--backend cuda` kernels (src/kernels/cuda/).  Builds the CLI with the CUDA
# kernels STATICALLY LINKED INTO zig-out/bin/qwen35 (nvcc -c -> nvcc -dlink ->
# addObjectFile + -lcudart/-lc++).  No kernel .so is produced or dlopen'd;
# libcudart is the only thing still opened at run time (src/cudart.zig).
cudabackend:
	$(ZIG) build -Dcuda=true -Doptimize=ReleaseFast $(ZIGCACHE)

# `make cuda` = rebuild the whole CLI with the CUDA backend linked in.
#
# It used to build a separate `libqwen35-cuda-kernels.so` kernel library, which
# the CLI loaded with dlopen.  That is gone: the .cu objects are now linked into
# the executable itself, so "building the CUDA backend" IS "rebuilding
# zig-out/bin/qwen35" and there is no separate artefact to depend on.
#
# The retired whole-model GPU forward (src/main_cuda.zig + src/kernel.cu +
# src/kernel_gdn.cu + src/kernel_matvec.cu) was deleted.  It was a PARALLEL
# engine forward that docs/cuda-backend-brief.md §6 retires -- it no longer
# compiled (main_cuda.zig called engine.Engine.load with 3 args while
# engine.zig:746 wants 5) and it shared no source with src/kernels/cuda/.
# `--backend cuda` on zig-out/bin/qwen35 is the only CUDA path.
#
# No .cu/.cuh prerequisites on purpose: build.zig registers every file in
# src/kernels/cuda/ as an explicit input of the nvcc steps, so Zig's own cache
# already re-runs nvcc when a kernel or a header changes.  Listing them again
# here would be a second, silently-drifting copy of that list.  Left .PHONY so
# zig always gets to decide what needs rebuilding; api.h is named only to
# document where the launcher ABI lives.
cuda: build.zig src/kernels/cuda/api.h
	$(ZIG) build -Dcuda=true -Doptimize=ReleaseFast $(ZIGCACHE)

# Compile-only check of every registered src/kernels/cuda/*.cu, no link, no
# install.  Use it to tell "build script wrong" from "kernel source wrong".
cuda-check:
	$(ZIG) build -Dcuda=true cuda-check $(ZIGCACHE) --summary all

# `--summary all` is REQUIRED for a trustworthy signal, not cosmetic.
#
# Zig 0.17.0 runs test artifacts over the `--listen=-` IPC protocol, and the
# handshake noise prints `failed command: ...test --listen=-` even when every
# test passes -- with exit code 0.  Reproduced on a bare 2-test project with no
# C code, so it is not this repo's bug.  Plain `zig build test` therefore prints
# a scary line while being green, and (worse) does not surface a REAL test
# failure in a way a scripted caller can rely on.
#
# `--summary all` fixes both: the last line is
#   Build Summary: 18/18 steps succeeded; 74/74 tests passed
# when green (exit 0), and
#   Build Summary: 16/18 steps succeeded (1 failed); 71/74 tests passed (3 failed)
# + `test transitive failure` when a test fails (exit 1).  Verified by
# perturbing a sign-table lookup in quant_dequant.zig: 3 parity tests caught it.
# The `failed command` line still prints; ignore it, read the summary.
test:
	$(ZIG) build test $(ZIGCACHE) --summary all

run0.8b: run0.8bq80 run0.8bbf16 run0.8biq4nl run0.8biq4xs run0.8bq3km run0.8bq3ks run0.8bq40 run0.8bq41 run0.8bq4km run0.8bq4ks run0.8bq5km run0.8bq5ks run0.8bq6k run0.8biq2m run0.8biq2xxs run0.8biq3xxs run0.8bq2kxl run0.8bq3kxl run0.8bq4kxl run0.8bq5kxl run0.8bq6kxl run0.8bq8kxl

run0.8b-cuda: run0.8bq80-cuda run0.8bbf16-cuda run0.8biq4nl-cuda run0.8biq4xs-cuda run0.8bq3km-cuda run0.8bq3ks-cuda run0.8bq40-cuda run0.8bq41-cuda run0.8bq4km-cuda run0.8bq4ks-cuda run0.8bq5km-cuda run0.8bq5ks-cuda run0.8bq6k-cuda run0.8biq2m-cuda run0.8biq2xxs-cuda run0.8biq3xxs-cuda run0.8bq2kxl-cuda run0.8bq3kxl-cuda run0.8bq4kxl-cuda run0.8bq5kxl-cuda run0.8bq6kxl-cuda run0.8bq8kxl-cuda

run0.8b-llamacpp: run0.8bq80-llamacpp run0.8bbf16-llamacpp run0.8biq4nl-llamacpp run0.8biq4xs-llamacpp run0.8bq3km-llamacpp run0.8bq3ks-llamacpp run0.8bq40-llamacpp run0.8bq41-llamacpp run0.8bq4km-llamacpp run0.8bq4ks-llamacpp run0.8bq5km-llamacpp run0.8bq5ks-llamacpp run0.8bq6k-llamacpp run0.8biq2m-llamacpp run0.8biq2xxs-llamacpp run0.8biq3xxs-llamacpp run0.8bq2kxl-llamacpp run0.8bq3kxl-llamacpp run0.8bq4kxl-llamacpp run0.8bq5kxl-llamacpp run0.8bq6kxl-llamacpp run0.8bq8kxl-llamacpp

run0.8b-llamacpp-cuda: run0.8bq80-llamacpp-cuda run0.8bbf16-llamacpp-cuda run0.8biq4nl-llamacpp-cuda run0.8biq4xs-llamacpp-cuda run0.8bq3km-llamacpp-cuda run0.8bq3ks-llamacpp-cuda run0.8bq40-llamacpp-cuda run0.8bq41-llamacpp-cuda run0.8bq4km-llamacpp-cuda run0.8bq4ks-llamacpp-cuda run0.8bq5km-llamacpp-cuda run0.8bq5ks-llamacpp-cuda run0.8bq6k-llamacpp-cuda run0.8biq2m-llamacpp-cuda run0.8biq2xxs-llamacpp-cuda run0.8biq3xxs-llamacpp-cuda run0.8bq2kxl-llamacpp-cuda run0.8bq3kxl-llamacpp-cuda run0.8bq4kxl-llamacpp-cuda run0.8bq5kxl-llamacpp-cuda run0.8bq6kxl-llamacpp-cuda run0.8bq8kxl-llamacpp-cuda

run0.8bq80:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 天空为什么是蓝色的？ -n 48 --threads 1

run0.8bq80-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda

run0.8bq80-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq80-llamacpp-cuda:
	/mnt/llm-dev/llama.cpp194/build/bin/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn -dev CUDA0
#	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn -dev Vulkan0

run0.8bbf16:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf 天空为什么是蓝色的？ -n 48 --threads 1

run0.8bbf16-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda

run0.8bbf16-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bbf16-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8biq4nl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_NL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8biq4nl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_NL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8biq4nl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_NL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8biq4nl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_NL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8biq4xs:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_XS.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8biq4xs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_XS.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8biq4xs-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_XS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8biq4xs-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-IQ4_XS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq3km:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq3km-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq3km-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq3km-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq3ks:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq3ks-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq3ks-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq3ks-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq40:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq40-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq40-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq40-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq41:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_1.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq41-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_1.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq41-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_1.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq41-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_1.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq4km:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq4km-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq4km-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq4km-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq4ks:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq4ks-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq4ks-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq4ks-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq5km:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq5km-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq5km-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq5km-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq5ks:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq5ks-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_S.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq5ks-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq5ks-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q5_K_S.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq6k:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q6_K.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq6k-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q6_K.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq6k-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q6_K.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq6k-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q6_K.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8biq2m:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8biq2m-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_M.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8biq2m-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8biq2m-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_M.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8biq2xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8biq2xxs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8biq2xxs-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8biq2xxs-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn


run0.8biq3xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8biq3xxs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8biq3xxs-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ3_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8biq3xxs-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ3_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq2kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq2kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq2kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq2kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q2_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq3kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q3_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq3kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q3_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq3kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q3_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq3kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q3_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq4kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq4kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq4kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq4kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q4_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq5kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q5_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq5kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q5_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq5kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q5_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq5kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q5_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq6kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q6_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq6kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q6_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq6kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q6_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq6kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q6_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run0.8bq8kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1
run0.8bq8kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf 天空为什么是蓝色的？ -n 48 --threads 1 --backend cuda


run0.8bq8kxl-llamacpp:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 -dev none --single-turn

run0.8bq8kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run2b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 天空为什么是蓝色的？

run2b-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run4b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 天空为什么是蓝色的？

# 27B on CPU: `-n` is the TOTAL sequence budget (prompt tokens + generated),
# not the token count — see engine.zig `while (pos < max_tokens)` starting at
# pos = prompt.len.  0.04-0.25 tok/s means an unbounded run (max_seq 4096)
# takes hours, so every 27B target carries an explicit -n.
run27bq80:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf 天空为什么是蓝色的？ -n 48

run27bq40:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf 天空为什么是蓝色的？ -n 48

run27bq40-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27biq3xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？ -n 48

run27biq3xxs-llamacpp-vulkan-long:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf -p "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" $(LONG_ARGS)

run27biq3xxs-llamacpp-cuda-long:
	/mnt/llm-dev/llama.cpp194/build/bin/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf -p "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" $(LONG_ARGS)

run27biq3xxs-llamacpp-vulkan:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run27biq3xxs-llamacpp-cuda:
	/mnt/llm-dev/llama.cpp194/build/bin/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run27biq3xxs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27biq3xxs-cuda-long:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" --backend cuda


test27b: run27biq3xxs-cuda-long run27biq3xxs-llamacpp-cuda-long run27biq3xxs-llamacpp-vulkan-long


run27biq2xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf 天空为什么是蓝色的？ -n 48

run27biq2xxs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27biq1s:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf 天空为什么是蓝色的？ -n 48

run27biq1s-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27biq4xs:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf 天空为什么是蓝色的？ -n 48

run27biq4xs-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27bq4kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf 天空为什么是蓝色的？ -n 48

run27bq4kxl-cuda:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf 天空为什么是蓝色的？ -n 48 --backend cuda

run27bq4kxl-llamacpp-cuda:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "天空为什么是蓝色的？" -n 48 -t 1 --single-turn

run27bq4kxl-llamacpp-cuda-long:
	/mnt/Qwen3.8-27B-GGUF/llama-server/llama-b10707/llama-cli -m /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf -p "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" $(LONG_ARGS)

run27bq8kl:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_L.gguf 天空为什么是蓝色的？ -n 48

run27bq8kxl:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf 天空为什么是蓝色的？ -n 48

# 27B 全量聚合（对齐 run0.8b 模式）：按 0.8b 顺序覆盖目录里全部 9 个量化。
run27b: run27bq80 run27bq40 run27biq4xs run27biq3xxs run27biq2xxs run27biq1s run27bq4kxl run27bq8kl run27bq8kxl

run27b-cuda: run27bq40-cuda run27biq4xs-cuda run27biq3xxs-cuda run27biq2xxs-cuda run27biq1s-cuda run27bq4kxl-cuda

run0.8b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" --backend cuda

run2b-cuda:
	CUDA_VISIBLE_DEVICES=0 zig-out/bin/qwen35 run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 天空为什么是蓝色的？ --backend cuda

run2b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 zig-out/bin/qwen35 run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" --backend cuda

run4b-cuda:
	CUDA_VISIBLE_DEVICES=0 zig-out/bin/qwen35 run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 天空为什么是蓝色的？ --backend cuda

run4b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 zig-out/bin/qwen35 run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。" --backend cuda
