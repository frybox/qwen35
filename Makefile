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
# Build cache on LOCAL disk: the project .zig-cache lives on NFS (/mnt/LLMs)
# and zig intermittently fails with "failed to write cache manifest: Unexpected"
# there. /tmp is local and fast.
ZIGCACHE := --cache-dir /tmp/qwen35-cache
HOST_OBJ := .zig-cache/qwen35_cuda_host.o
CUDA_EXE := zig-out/bin/qwen35-cuda

.PHONY: all build cpu cpufast cuda test run0.8b run2b run4b run27bq8 run27bq4 run27biq3xxs run27biq2xxs run27biq1s run27b-long run0.8b-cuda run2b-cuda run4b-cuda

all: build cpu cuda

build:
	$(ZIG) build $(ZIGCACHE)

cpu:
	$(ZIG) build $(ZIGCACHE)

cpufast:
	$(ZIG) build -Doptimize=ReleaseFast $(ZIGCACHE)

cuda: $(CUDA_EXE)

$(CUDA_EXE): $(HOST_OBJ) src/kernel.cu
	$(NVCC) -O3 -arch=native -o=$@ $(HOST_OBJ) src/kernel.cu -lcudart -lm

$(HOST_OBJ): src/main_cuda.zig src/engine.zig src/tokenizer.zig src/sampler.zig
	mkdir -p $(dir $@)
	$(ZIG) cc -c $< -o $@

test:
	$(ZIG) build test $(ZIGCACHE)

run0.8b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 天空为什么是蓝色的？

run2b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 天空为什么是蓝色的？

run4b:
	zig-out/bin/qwen35 run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 天空为什么是蓝色的？

# 27B on CPU: `-n` is the TOTAL sequence budget (prompt tokens + generated),
# not the token count — see engine.zig `while (pos < max_tokens)` starting at
# pos = prompt.len.  0.04-0.25 tok/s means an unbounded run (max_seq 4096)
# takes hours, so every 27B target carries an explicit -n.
run27bq8:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf 天空为什么是蓝色的？ -n 48

run27bq4:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf 天空为什么是蓝色的？ -n 48

run27biq3xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？ -n 48

run27biq3xxs-cuda:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf 天空为什么是蓝色的？

run27biq2xxs:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf 天空为什么是蓝色的？ -n 48

run27biq1s:
	zig-out/bin/qwen35 run /mnt/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf 天空为什么是蓝色的？ -n 48

run0.8b-cuda:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 天空为什么是蓝色的？

run0.8b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。"

run2b-cuda:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 天空为什么是蓝色的？

run2b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。"

run4b-cuda:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf 天空为什么是蓝色的？

run4b-cuda-long:
	CUDA_VISIBLE_DEVICES=0 $(CUDA_EXE) run /mnt/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf "茶起源于中国，已有数千年的饮用历史。最早的茶是被当作药物使用的，神农尝百草的传说中就有茶的身影。后来人们发现茶可以解渴去乏，逐渐从药物变成了日常饮品。唐代陆羽写下一部茶经，系统记录了茶的种植、采摘、烹煮和饮用方法，被后人尊为茶圣。宋代点茶风靡一时，人们在茶汤上击拂出细腻泡沫，再在泡沫上作画，称为茶百戏。到了明清时期，散茶冲泡取代了团饼煎茶，紫砂壶与盖碗成为主要器具，工夫茶与盖碗茶等冲泡方式由此定型。 中国茶类繁多，按发酵程度可分为绿茶、白茶、黄茶、青茶、红茶和黑茶。绿茶不发酵，保留了鲜叶中 的大部分营养成分，汤色清亮，滋味鲜爽，西湖龙井、碧螺春、黄山毛峰都是名品。白茶不炒不揉，自然萎凋，滋味清淡回甘，白毫银针是其中的代表。青茶即乌龙茶，半发酵，既有绿茶的清香又有红茶的醇厚，铁观音、大红袍、凤凰单丛都属此类。红茶全发酵，汤色红艳，滋味甜醇，祁门红茶以独特的香气闻名。黑茶属后发酵，陈化后滋味醇厚，普洱熟茶便是典型。 泡茶讲究水温、时间和器具。嫩绿茶用八十度左右的水，避免烫出苦涩；乌龙茶用沸 水，高香高泡；红茶用九十五度的水，让内含物质充分释放。杯泡、壶泡、盖碗冲泡各有韵味，盖碗最考验手法，烫盏、投茶、注水、出汤一气呵成，方能泡出一杯干净纯正的茶。茶席之上，一盏清茶，几本闲书，慢煮光阴，是许多人向往的生活状态。 茶与饮食同样相配。饭后一杯浓茶消食解腻，午后 一盏清茶提神醒脑，深夜一盏老白茶温润养胃。茶席配茶点，桂花糕、绿豆糕、坚果蜜饯，甜而不腻，与茶汤相映成趣。 茶山多在高海拔云雾缭绕之处 ，昼夜温差大，茶树生长缓慢，内含物质积累丰富，因而滋味饱满耐泡，常能冲泡十余道而不减其韵。据此给我写篇不低于600字的作文。"
