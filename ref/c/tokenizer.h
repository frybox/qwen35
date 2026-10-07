#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

// 特殊 token 定义 (Qwen3.5: 见 tokenizer_config.json added_tokens_decoder)
#define SPECIAL_TOKEN_EOS 248046   // <|im_end|>
#define SPECIAL_TOKEN_PAD 248044   // <|endoftext|>

// BPE Tokenizer 结构
typedef struct {
    // 词表: token string -> id
    char** vocab;           // id -> string 映射
    int vocab_size;
    
    // 合并规则
    struct {
        char* left;
        char* right;
        char* merged;       // 合并后的字符串
    }* merges;
    int num_merges;
    
    // 特殊 token 映射
    struct {
        char* token;
        int id;
    }* special_tokens;
    int num_special_tokens;
    
    // 内部缓存
    void* cache;            // 用于加速编码
} BPETokenizer;

// ============ API ============

// 初始化和释放
int tokenizer_init(BPETokenizer* tok);
void tokenizer_free(BPETokenizer* tok);

// 加载词表和合并规则
int tokenizer_load_vocab(BPETokenizer* tok, const char* vocab_path);
int tokenizer_load_merges(BPETokenizer* tok, const char* merges_path);
int tokenizer_load_special_tokens(BPETokenizer* tok);

// 编码: 文本 -> token ids
// 返回 token 数量，-1 表示错误
int tokenizer_encode(BPETokenizer* tok, const char* text, int* token_ids, int max_tokens);

// 解码: token ids -> 文本
// 返回解码后的文本长度，-1 表示错误
int tokenizer_decode(BPETokenizer* tok, const int* token_ids, int num_tokens, char* output, int max_output);

// 辅助函数
const char* tokenizer_get_token(BPETokenizer* tok, int id);
int tokenizer_get_id(BPETokenizer* tok, const char* token);

// 测试函数
void test_tokenizer(const char* vocab_path, const char* merges_path);

#endif // TOKENIZER_H