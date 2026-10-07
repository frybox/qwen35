CC = gcc
CFLAGS = -Wall -Wextra -O2 -g

MODEL_DIR ?= /mnt/Qwen3.5-0.8B

all: qwen35_engine

qwen35_engine: qwen35_engine.c safetensors.c safetensors.h tokenizer.c tokenizer.h nfc_table.h
	$(CC) $(CFLAGS) -o qwen35_engine qwen35_engine.c safetensors.c tokenizer.c -lm

run: qwen35_engine
	./qwen35_engine "$(PROMPT)"

clean:
	rm -f qwen35_engine

.PHONY: all run clean
