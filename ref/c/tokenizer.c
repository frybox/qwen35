#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 1024   // 最长token: 256个字节, 映射成GPT-2字符后最多768字节UTF-8
#define MAX_LINE_LEN 2048
#define MAX_VOCAB_SIZE 260000
#define MAX_MERGES 300000

// 哈希表用于 token string -> id 映射
#define HASH_SIZE 524287  // 大质数

typedef struct HashEntry {
    char* key;
    int value;
    struct HashEntry* next;
} HashEntry;

typedef struct {
    HashEntry* table[HASH_SIZE];
} HashMap;

// 哈希函数
static unsigned int hash_string(const char* str) {
    unsigned int h = 5381;
    while (*str) {
        h = ((h << 5) + h) + (unsigned char)(*str++);
    }
    return h % HASH_SIZE;
}

static void hashmap_init(HashMap* map) {
    memset(map->table, 0, sizeof(map->table));
}

static void hashmap_put(HashMap* map, const char* key, int value) {
    unsigned int idx = hash_string(key);
    HashEntry* entry = map->table[idx];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    entry = (HashEntry*)malloc(sizeof(HashEntry));
    entry->key = strdup(key);
    entry->value = value;
    entry->next = map->table[idx];
    map->table[idx] = entry;
}

static int hashmap_get(HashMap* map, const char* key, int* found) {
    unsigned int idx = hash_string(key);
    HashEntry* entry = map->table[idx];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            *found = 1;
            return entry->value;
        }
        entry = entry->next;
    }
    *found = 0;
    return -1;
}

static void hashmap_free(HashMap* map) {
    for (int i = 0; i < HASH_SIZE; i++) {
        HashEntry* entry = map->table[i];
        while (entry) {
            HashEntry* next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
}

// 内部结构
typedef struct {
    HashMap token_to_id;    // string -> id
    HashMap merge_rank;     // 合并规则优先级
    char byte_encoder[256][8];  // 字节 -> GPT-2 字符映射
    int byte_to_token_id[256];  // 字节 -> token ID
    int unicode_to_byte[65536]; // Unicode -> 原始字节映射（用于解码）
} TokenizerInternal;

// GPT-2 字节到字符映射表 (正确版本)
// 可打印 ASCII (! ~) 和 Latin-1 扩展 (¡ ¬, ® ÿ) 保持不变
// 其他字节映射到 U+0100 开始
static const int gpt2_byte_to_unicode[256] = {
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,  // 0x00-0x07
    0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x010D, 0x010E, 0x010F,  // 0x08-0x0F
    0x0110, 0x0111, 0x0112, 0x0113, 0x0114, 0x0115, 0x0116, 0x0117,  // 0x10-0x17
    0x0118, 0x0119, 0x011A, 0x011B, 0x011C, 0x011D, 0x011E, 0x011F,  // 0x18-0x1F
    0x0120,  // 0x20 (空格 -> Ġ)
    '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',  // 0x21-0x2F
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',  // 0x30-0x3F
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',  // 0x40-0x4F
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',  // 0x50-0x5F
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',  // 0x60-0x6F
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',      // 0x70-0x7E
    0x0121,  // 0x7F (DEL -> ġ)
    0x0122, 0x0123, 0x0124, 0x0125, 0x0126, 0x0127, 0x0128, 0x0129,  // 0x80-0x87
    0x012A, 0x012B, 0x012C, 0x012D, 0x012E, 0x012F, 0x0130, 0x0131,  // 0x88-0x8F
    0x0132, 0x0133, 0x0134, 0x0135, 0x0136, 0x0137, 0x0138, 0x0139,  // 0x90-0x97
    0x013A, 0x013B, 0x013C, 0x013D, 0x013E, 0x013F, 0x0140, 0x0141,  // 0x98-0x9F
    0x0142,  // 0xA0 (NBSP -> ł)
    0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC,  // 0xA1-0xAC (¡ ¬)
    0x0143,  // 0xAD (soft hyphen -> Ń)
    0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9,  // 0xAE-0xB9 (® ¹)
    0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,  // 0xBA-0xBF (º ¿)
    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,  // 0xC0-0xC7 (À Ç)
    0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF,  // 0xC8-0xCF (È Ï)
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,  // 0xD0-0xD7 (Ð ×)
    0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF,  // 0xD8-0xDF (Ø ß)
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,  // 0xE0-0xE7 (à ç)
    0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,  // 0xE8-0xEF (è ï)
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,  // 0xF0-0xF7 (ð ÷)
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF   // 0xF8-0xFF (ø ÿ)
};

// Unicode 码点到 UTF-8
static int unicode_to_utf8(int codepoint, char* out) {
    if (codepoint < 0x80) {
        out[0] = codepoint;
        return 1;
    } else if (codepoint < 0x800) {
        out[0] = 0xC0 | (codepoint >> 6);
        out[1] = 0x80 | (codepoint & 0x3F);
        return 2;
    } else if (codepoint < 0x10000) {
        out[0] = 0xE0 | (codepoint >> 12);
        out[1] = 0x80 | ((codepoint >> 6) & 0x3F);
        out[2] = 0x80 | (codepoint & 0x3F);
        return 3;
    } else {
        out[0] = 0xF0 | (codepoint >> 18);
        out[1] = 0x80 | ((codepoint >> 12) & 0x3F);
        out[2] = 0x80 | ((codepoint >> 6) & 0x3F);
        out[3] = 0x80 | (codepoint & 0x3F);
        return 4;
    }
}

// UTF-8 处理函数
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// 读取 UTF-8 字符的 Unicode 码点
static int utf8_to_unicode(const char* str, int* char_len) {
    unsigned char c = (unsigned char)str[0];
    if (c < 0x80) {
        *char_len = 1;
        return c;
    } else if ((c & 0xE0) == 0xC0) {
        *char_len = 2;
        return ((c & 0x1F) << 6) | (str[1] & 0x3F);
    } else if ((c & 0xF0) == 0xE0) {
        *char_len = 3;
        return ((c & 0x0F) << 12) | ((str[1] & 0x3F) << 6) | (str[2] & 0x3F);
    } else if ((c & 0xF8) == 0xF0) {
        *char_len = 4;
        return ((c & 0x07) << 18) | ((str[1] & 0x3F) << 12) | ((str[2] & 0x3F) << 6) | (str[3] & 0x3F);
    }
    *char_len = 1;
    return c;
}

// 初始化字节编码器
static void init_byte_encoder(TokenizerInternal* internal) {
    // 初始化 Unicode 到字节的映射表
    for (int i = 0; i < 65536; i++) {
        internal->unicode_to_byte[i] = -1;
    }
    
    for (int b = 0; b < 256; b++) {
        int unicode = gpt2_byte_to_unicode[b];
        int len = unicode_to_utf8(unicode, internal->byte_encoder[b]);
        internal->byte_encoder[b][len] = '\0';
        internal->byte_to_token_id[b] = -1;
        internal->unicode_to_byte[unicode] = b;  // 建立 Unicode 到字节的反向映射
    }
}

// 解析 JSON 字符串 (简化版，针对 vocab.json 格式)
static char* parse_json_string(const char* start, const char** end) {
    if (*start != '"') return NULL;
    start++;
    
    char* result = (char*)malloc(MAX_TOKEN_LEN);
    int len = 0;
    
    while (*start && *start != '"' && len < MAX_TOKEN_LEN - 1) {
        if (*start == '\\') {
            start++;
            switch (*start) {
                case 'n': result[len++] = '\n'; break;
                case 't': result[len++] = '\t'; break;
                case 'r': result[len++] = '\r'; break;
                case '\\': result[len++] = '\\'; break;
                case '"': result[len++] = '"'; break;
                case 'u': {
                    // Unicode escape: \uXXXX
                    start++;
                    unsigned int code = 0;
                    for (int i = 0; i < 4 && isxdigit((unsigned char)start[i]); i++) {
                        code = code * 16;
                        if (isdigit((unsigned char)start[i])) code += start[i] - '0';
                        else code += (tolower(start[i]) - 'a' + 10);
                    }
                    start += 3;  // 循环会再+1
                    // 转换为 UTF-8
                    len += unicode_to_utf8(code, result + len);
                    break;
                }
                default: result[len++] = *start; break;
            }
        } else {
            result[len++] = *start;
        }
        start++;
    }
    result[len] = '\0';
    if (*start == '"') start++;
    *end = start;
    return result;
}

// 初始化
int tokenizer_init(BPETokenizer* tok) {
    memset(tok, 0, sizeof(BPETokenizer));
    
    tok->vocab = (char**)calloc(MAX_VOCAB_SIZE, sizeof(char*));
    tok->merges = (void*)calloc(MAX_MERGES, sizeof(*tok->merges));
    tok->special_tokens = (void*)calloc(100, sizeof(*tok->special_tokens));
    
    if (!tok->vocab || !tok->merges || !tok->special_tokens) {
        return -1;
    }
    
    // 初始化内部哈希表
    TokenizerInternal* internal = (TokenizerInternal*)malloc(sizeof(TokenizerInternal));
    hashmap_init(&internal->token_to_id);
    hashmap_init(&internal->merge_rank);
    init_byte_encoder(internal);
    tok->cache = internal;
    
    return 0;
}

// 释放
void tokenizer_free(BPETokenizer* tok) {
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    
    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            free(tok->merges[i].left);
            free(tok->merges[i].right);
            free(tok->merges[i].merged);
        }
        free(tok->merges);
    }
    
    if (tok->special_tokens) {
        for (int i = 0; i < tok->num_special_tokens; i++) {
            free(tok->special_tokens[i].token);
        }
        free(tok->special_tokens);
    }
    
    if (tok->cache) {
        TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
        hashmap_free(&internal->token_to_id);
        hashmap_free(&internal->merge_rank);
        free(internal);
    }
}

// 加载 vocab.json
int tokenizer_load_vocab(BPETokenizer* tok, const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "无法打开词表文件: %s\n", vocab_path);
        return -1;
    }
    
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    
    char line[MAX_LINE_LEN];
    int id = 0;
    
    // 读取第一行（应该包含 {）
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }
    
    // 逐行解析
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        // 跳过空白
        while (*p && isspace(*p)) p++;
        
        if (*p == '}') break;  // 结束
        if (*p != '"') continue;
        
        // 解析 token 字符串
        const char* end;
        char* token = parse_json_string(p, &end);
        if (!token) continue;
        
        // 跳过冒号
        p = (char*)end;
        while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
        
        // 解析 ID
        id = atoi(p);
        
        // 存储
        if (id >= 0 && id < MAX_VOCAB_SIZE) {
            if (tok->vocab[id]) free(tok->vocab[id]);
            tok->vocab[id] = token;
            if (id >= tok->vocab_size) tok->vocab_size = id + 1;
            
            // 同时存入哈希表
            hashmap_put(&internal->token_to_id, token, id);
            
            // 检查是否是单字节 token，更新字节映射
            if (strlen(token) <= 4) {
                int char_len;
                int unicode = utf8_to_unicode(token, &char_len);
                if (char_len == (int)strlen(token)) {
                    // 找到对应的字节
                    for (int b = 0; b < 256; b++) {
                        if (gpt2_byte_to_unicode[b] == unicode) {
                            internal->byte_to_token_id[b] = id;
                            break;
                        }
                    }
                }
            }
        } else {
            free(token);
        }
    }
    
    fclose(f);
    printf("加载词表完成: %d 个 token\n", tok->vocab_size);
    return 0;
}

// 加载 merges.txt
int tokenizer_load_merges(BPETokenizer* tok, const char* merges_path) {
    FILE* f = fopen(merges_path, "r");
    if (!f) {
        fprintf(stderr, "无法打开合并规则文件: %s\n", merges_path);
        return -1;
    }
    
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    
    char line[MAX_LINE_LEN];
    int rank = 0;
    
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        // 注意: merges.txt 首行是 "#version: ..." 才是注释头,
        // 文件中间存在以 '#' 开头的合法 merge(如 "# #", "# include", "# define"),
        // 只有"没有空格分隔"的 # 行才当注释跳过(下方找不到空格时 continue 已覆盖)。
        // 找到分隔空格
        char* space = strchr(p, ' ');
        if (!space) continue;
        
        // 提取 left
        int left_len = space - p;
        char* left = (char*)malloc(left_len + 1);
        strncpy(left, p, left_len);
        left[left_len] = '\0';
        
        // 找到 right (到换行或结尾)
        p = space + 1;
        char* end = p;
        while (*end && *end != '\n' && *end != '\r') end++;
        int right_len = end - p;
        
        char* right = (char*)malloc(right_len + 1);
        strncpy(right, p, right_len);
        right[right_len] = '\0';
        
        // 创建合并后的字符串
        char* merged = (char*)malloc(left_len + right_len + 1);
        strcpy(merged, left);
        strcat(merged, right);
        
        // 存储
        int idx = tok->num_merges;
        tok->merges[idx].left = left;
        tok->merges[idx].right = right;
        tok->merges[idx].merged = merged;
        tok->num_merges++;
        
        // 存储优先级 (用于 BPE)
        char merge_key[MAX_TOKEN_LEN * 2];
        snprintf(merge_key, sizeof(merge_key), "%s %s", left, right);
        hashmap_put(&internal->merge_rank, merge_key, rank);
        rank++;
    }
    
    fclose(f);
    printf("加载合并规则完成: %d 条\n", tok->num_merges);
    return 0;
}

// 加载特殊 token
// 注意: id 必须与 tokenizer_config.json 的 added_tokens_decoder 一致 (Qwen3.5 从 248044 起,
// 不是 Qwen2.5 时代的 151643+; vocab.json 中不含这些 token)
int tokenizer_load_special_tokens(BPETokenizer* tok) {
    struct { const char* token; int id; } special[] = {
        {"<|endoftext|>", 248044},
        {"<|im_start|>", 248045},
        {"<|im_end|>", 248046},
        {"<|object_ref_start|>", 248047},
        {"<|object_ref_end|>", 248048},
        {"<|box_start|>", 248049},
        {"<|box_end|>", 248050},
        {"<|quad_start|>", 248051},
        {"<|quad_end|>", 248052},
        {"<|vision_start|>", 248053},
        {"<|vision_end|>", 248054},
        {"<|vision_pad|>", 248055},
        {"<|image_pad|>", 248056},
        {"<|video_pad|>", 248057},
        {"<tool_call>", 248058},
        {"</tool_call>", 248059},
        {"<|fim_prefix|>", 248060},
        {"<|fim_middle|>", 248061},
        {"<|fim_suffix|>", 248062},
        {"<|fim_pad|>", 248063},
        {"<|repo_name|>", 248064},
        {"<|file_sep|>", 248065},
        {"<tool_response>", 248066},
        {"</tool_response>", 248067},
        {"<think>", 248068},
        {"</think>", 248069},
        {"<|audio_start|>", 248070},
        {"<|audio_end|>", 248071},
        {NULL, 0}
    };
    
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    
    for (int i = 0; special[i].token; i++) {
        int idx = tok->num_special_tokens;
        tok->special_tokens[idx].token = strdup(special[i].token);
        tok->special_tokens[idx].id = special[i].id;
        tok->num_special_tokens++;
        
        // 同时加入词表（覆盖原有内容，因为特殊 token 不在 vocab.json 中）
        int id = special[i].id;
        if (id >= tok->vocab_size) tok->vocab_size = id + 1;
        if (tok->vocab[id]) free(tok->vocab[id]);  // 覆盖原有的（vocab.json 中的占位内容）
        tok->vocab[id] = strdup(special[i].token);
        hashmap_put(&internal->token_to_id, special[i].token, id);
    }
    
    printf("加载特殊 token 完成: %d 个\n", tok->num_special_tokens);
    return 0;
}

// 检查是否是特殊 token
static int is_special_token(BPETokenizer* tok, const char* text, int* id, int* len) {
    for (int i = 0; i < tok->num_special_tokens; i++) {
        const char* token = tok->special_tokens[i].token;
        int token_len = strlen(token);
        if (strncmp(text, token, token_len) == 0) {
            *id = tok->special_tokens[i].id;
            *len = token_len;
            return 1;
        }
    }
    return 0;
}

// BPE 编码核心：将字节序列转换为 token
// parts 数组动态分配，支持任意长度输入（无 256 硬截断）
static int bpe_encode_bytes(BPETokenizer* tok, const unsigned char* bytes, int num_bytes, int* tokens, int max_tokens) {
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    
    if (num_bytes <= 0) return 0;
    
    // 初始化：每个字节对应一个 GPT-2 字符（动态数组）
    int capacity = num_bytes + 1;
    char** parts = (char**)malloc(capacity * sizeof(char*));
    if (!parts) return 0;
    int num_parts = 0;
    
    for (int i = 0; i < num_bytes; i++) {
        parts[num_parts++] = strdup(internal->byte_encoder[bytes[i]]);
    }
    
    // BPE 合并循环
    while (num_parts > 1) {
        // 找到优先级最高的合并
        int best_rank = -1;
        int best_idx = -1;
        
        for (int i = 0; i < num_parts - 1; i++) {
            char merge_key[MAX_TOKEN_LEN * 2 + 2];  // 最长token映射后768B, 拼key翻倍+空格
            int key_len = strlen(parts[i]) + strlen(parts[i + 1]) + 2;
            if (key_len > (int)sizeof(merge_key)) { continue; }  // 超长key不可能在merges表里(最长512B+1+512B>1026? 512+1+512=1025 <= 1026 OK)
            snprintf(merge_key, sizeof(merge_key), "%s %s", parts[i], parts[i + 1]);
            
            int found = 0;
            int rank = hashmap_get(&internal->merge_rank, merge_key, &found);
            if (found && (best_rank < 0 || rank < best_rank)) {
                best_rank = rank;
                best_idx = i;
            }
        }
        
        if (best_idx < 0) break;  // 没有可合并的了
        
        // 执行合并
        char* merged = (char*)malloc(strlen(parts[best_idx]) + strlen(parts[best_idx + 1]) + 1);
        strcpy(merged, parts[best_idx]);
        strcat(merged, parts[best_idx + 1]);
        
        free(parts[best_idx]);
        free(parts[best_idx + 1]);
        parts[best_idx] = merged;
        
        // 移动后面的部分
        for (int i = best_idx + 1; i < num_parts - 1; i++) {
            parts[i] = parts[i + 1];
        }
        num_parts--;
    }
    
    // 转换为 token ID
    int num_tokens = 0;
    for (int i = 0; i < num_parts && num_tokens < max_tokens; i++) {
        int found = 0;
        int id = hashmap_get(&internal->token_to_id, parts[i], &found);
        if (found) {
            tokens[num_tokens++] = id;
        } else {
            // 未知 token，尝试逐字节
            for (int j = 0; parts[i][j] && num_tokens < max_tokens; ) {
                int char_len = utf8_char_len((unsigned char)parts[i][j]);
                char single[8];
                strncpy(single, parts[i] + j, char_len);
                single[char_len] = '\0';
                id = hashmap_get(&internal->token_to_id, single, &found);
                if (found) {
                    tokens[num_tokens++] = id;
                }
                j += char_len;
            }
        }
        free(parts[i]);
    }
    
    free(parts);
    return num_tokens;
}

// ==================== NFC 归一化 ====================
// tokenizer.json 的 normalizer 是 NFC。HF 在 pre_tokenizer 之前做归一化,
// 影响: e+U+0301 -> é (1 token), CJK 兼容字 U+2F960 -> U+4202 等。
// 实现: 单码点替换表 + canonical composition pair 表 + Hangul 算法组合。
#include "nfc_table.h"

// 二分查表
static int nfc_lookup_pair(int starter, int combining) {
    int lo = 0, hi = nfc_pairs_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (nfc_pairs[mid][0] < starter) lo = mid + 1;
        else if (nfc_pairs[mid][0] > starter) hi = mid - 1;
        else if (nfc_pairs[mid][1] < combining) lo = mid + 1;
        else if (nfc_pairs[mid][1] > combining) hi = mid - 1;
        else return nfc_pairs[mid][2];
    }
    return 0;
}

static int nfc_lookup_single(int cp) {
    int lo = 0, hi = nfc_single_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (nfc_single[mid][0] < cp) lo = mid + 1;
        else if (nfc_single[mid][0] > cp) hi = mid - 1;
        else return nfc_single[mid][1];
    }
    return 0;
}

// 组合字符的 canonical combining class (0 = 非组合/starter)
// 完整 CCC 表太大; 组合对表里的 combining 字符可以任意序(规范排序只影响组合机会),
// 实用实现: 只处理"紧邻组合" — 大部分真实文本(分解的 é, ä 等)组合符紧跟 starter。
static int is_combining_mark(int cp) {
    // 常见组合区间 (U+0300-U+036F, 一部分扩展区)
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x0483 && cp <= 0x0489) ||
           (cp >= 0x0591 && cp <= 0x05BD) ||
           (cp >= 0x0610 && cp <= 0x061A) ||
           (cp >= 0x064B && cp <= 0x065F) ||
           (cp >= 0x0670 && cp == 0x0670) ||
           (cp >= 0x06D6 && cp <= 0x06DC) ||
           (cp >= 0x0900 && cp <= 0x0903) ||
           (cp >= 0x093A && cp <= 0x094F) ||
           (cp >= 0x0951 && cp <= 0x0957) ||
           (cp >= 0x0E31 && cp <= 0x0E3A) ||
           (cp >= 0x0E47 && cp <= 0x0E4E) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20F0) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

// 对文本做 NFC 归一化, 结果写入 out (需 >= 4*len+1 字节), 返回输出字节长度
int nfc_normalize(const char* text, char* out, int out_cap) {
    int slen = (int)strlen(text);
    int cap = slen + 16;                    // 每个输入字符最多产生少量码点, 上限即字符数
    int* cps = (int*)malloc(cap * sizeof(int));
    if (!cps) { out[0] = '\0'; if (out_cap>0) out[0]='\0'; return 0; }
    int n = 0;
    const unsigned char* p = (const unsigned char*)text;
    // 1) decode + 单码点替换 + Hangul 分解到可组合态(LV->L+V 等)
    while (*p && n < cap - 4) {
        int cl;
        int cp = utf8_to_unicode((const char*)p, &cl);
        p += cl;
        int repl = nfc_lookup_single(cp);
        if (repl) cp = repl;
        cps[n++] = cp;
    }
    // 2) 递归组合: 从左往右, 尝试 starter+combining 合成, 可合成则替换后继续尝试与前一个再合成
    int m = 0;  // 输出栈
    for (int i = 0; i < n; i++) {
        int cp = cps[i];
        if (m > 0 && is_combining_mark(cp)) {
            // 向前找最近的 starter(跳过无法参与组合的非组合字符)
            int j = m - 1;
            int composed = 0;
            // 先与栈顶直接组合, 若栈顶本身是组合符, 先合成它下面那层
            while (j >= 0) {
                if (!is_combining_mark(cps[j])) {
                    composed = nfc_lookup_pair(cps[j], cp);
                    if (composed) { cps[j] = composed; cp = 0; }
                    break;
                }
                j--;
            }
            if (!cp) continue;  // 已合成
        }
        cps[m++] = cp;
    }
    // Hangul: 组合对表已含 L+V, LV+T 等条目(unicodedata 导出), 无需特殊算法
    // 3) encode 回 UTF-8
    int out_len = 0;
    for (int i = 0; i < m && out_len < out_cap - 5; i++) {
        out_len += unicode_to_utf8(cps[i], out + out_len);
    }
    out[out_len] = '\0';
    free(cps);
    return out_len;
}

// ==================== end NFC ====================

// ==================== pre_tokenizer ====================
// 与 HF tokenizer.json 中 pre_tokenizer 正则对齐 (GPT-2 风格):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   \p{N}
//    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   \s*[\r\n]+
//   \s+(?!\S)
//   \s+
// 实现方式: 对每个 UTF-8 字符分类 (字母/数字/空白/换行/其他),
// 手写状态机按上述规则切出一个个"预分词片段", 再逐段做 BPE。

typedef enum { CL_LETTER, CL_NUMBER, CL_SPACE, CL_NEWLINE, CL_OTHER } CharClass;

static CharClass classify_char(int cp) {
    if (cp == '\r' || cp == '\n') return CL_NEWLINE;
    if (cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f') return CL_SPACE;
    // ASCII 快速路径
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CL_LETTER;
        if (cp >= '0' && cp <= '9') return CL_NUMBER;
        return CL_OTHER;
    }
    // \p{L}: 所有 Unicode 字母 (含 CJK)
    if ((cp >= 0x00C0 && cp <= 0x024F) ||   // Latin Extended
        (cp >= 0x0370 && cp <= 0x04FF) ||   // Greek/Cyrillic
        (cp >= 0x0590 && cp <= 0x06FF) ||   // Hebrew/Arabic
        (cp >= 0x0900 && cp <= 0x0DFF) ||   // 印度语系
        (cp >= 0x0E00 && cp <= 0x0EFF) ||   // 泰文
        (cp >= 0x1000 && cp <= 0x109F) ||   // 缅文
        (cp >= 0x1100 && cp <= 0x11FF) ||   // 谚文
        (cp >= 0x1E00 && cp <= 0x1FFF) ||   // 拉丁扩展附加/希腊扩展
        (cp >= 0x2C60 && cp <= 0x2CDF) ||   // 拉丁扩展-C
        (cp >= 0x3040 && cp <= 0x30FF) ||   // 平假名/片假名
        (cp >= 0x3130 && cp <= 0x318F) ||   // 谚文兼容
        (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK 扩展 A
        (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK 基本区
        (cp >= 0xA000 && cp <= 0xA4CF) ||   // 彝文
        (cp >= 0xAC00 && cp <= 0xD7AF) ||   // 谚文音节
        (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK 兼容
        (cp >= 0x1E000 && cp <= 0x1EFFF) || // 古代文字等
        (cp >= 0x20000 && cp <= 0x2FA1F))   // CJK 扩展 B-F
        return CL_LETTER;
    // \p{M}: 组合附加符号 (Pinyin 声调、梵文 matra 等)
    if ((cp >= 0x0300 && cp <= 0x036F) ||   // 组合变音符号
        (cp >= 0x0483 && cp <= 0x0489) ||
        (cp >= 0x0591 && cp <= 0x05BD) ||
        (cp >= 0x0610 && cp <= 0x061A) ||
        (cp >= 0x064B && cp <= 0x065F) ||
        (cp >= 0x0670 && cp <= 0x0670) ||
        (cp >= 0x06D6 && cp <= 0x06DC) ||
        (cp >= 0x0900 && cp <= 0x0903) ||
        (cp >= 0x093A && cp <= 0x094F) ||
        (cp >= 0x0951 && cp <= 0x0957) ||
        (cp >= 0x0E31 && cp <= 0x0E3A) ||
        (cp >= 0x0E47 && cp <= 0x0E4E) ||
        (cp >= 0x1AB0 && cp <= 0x1AFF) ||
        (cp >= 0x1DC0 && cp <= 0x1DFF) ||
        (cp >= 0x20D0 && cp <= 0x20F0) ||
        (cp >= 0xFE00 && cp <= 0xFE0F) ||   // 变体选择符
        (cp >= 0xFE20 && cp <= 0xFE2F))
        return CL_LETTER;
    // \p{N}: 数字 (含全角)
    if ((cp >= 0x0660 && cp <= 0x0669) ||   // 阿拉伯-印度数字
        (cp >= 0x06F0 && cp <= 0x06F9) ||
        (cp >= 0x0966 && cp <= 0x096F) ||   // 梵文数字
        (cp >= 0x09E6 && cp <= 0x09EF) ||
        (cp >= 0x0A66 && cp <= 0x0A6F) ||
        (cp >= 0x0BE6 && cp <= 0x0BEF) ||
        (cp >= 0x0C66 && cp <= 0x0C6F) ||
        (cp >= 0x0E50 && cp <= 0x0E59) ||   // 泰文数字
        (cp >= 0xFF10 && cp <= 0xFF19))     // 全角数字
        return CL_NUMBER;
    // 其他空白 (\p{Z} 及 Unicode 空白)
    if (cp == 0x00A0 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000 || cp == 0x1680)
        return CL_SPACE;
    return CL_OTHER;
}

// 空白分支: 扫描 [0..q) 的完整空白段; look_ahead_special 非空表示
// 空白段结束后紧跟特殊 token —— 此时 HF 行为是"段边界" 整段空白按 \s+(?!\S) 全取
// (不加 \s*[\r\n]+ 贪婪到换行团, 而是先尝试空白团整段, 因为 special 是独立的下一片段)
static int scan_space_run(const unsigned char* p, int c0_len, int* q_out, int* nl_end_out) {
    int q = c0_len;
    int last_nl_end = -1;
    while (p[q]) {
        int cl;
        int c = utf8_to_unicode((const char*)(p + q), &cl);
        CharClass cc = classify_char(c);
        if (cc == CL_NEWLINE) { last_nl_end = q + cl; q += cl; }
        else if (cc == CL_SPACE) { q += cl; }
        else break;
    }
    *q_out = q;
    *nl_end_out = last_nl_end;
    return 0;
}

// 从 text 的当前位置切出一个预分词片段
// 返回片段字节长度, *is_special_out 表明是否命中特殊 token (此时直接返回其长度)
static int next_pretoken(BPETokenizer* tok, const char* text, int* is_special_out, int* special_id_out) {
    // 特殊 token 优先 (encode 主循环在遇到时直接整段输出)
    int sp_id, sp_len;
    if (is_special_token(tok, text, &sp_id, &sp_len)) {
        *is_special_out = 1;
        *special_id_out = sp_id;
        return sp_len;
    }
    *is_special_out = 0;
    *special_id_out = -1;
    if (*text == '\0') return 0;

    const unsigned char* p = (const unsigned char*)text;
    int c0_len;
    int c0 = utf8_to_unicode((const char*)p, &c0_len);

    // 分支1: 缩写 's 't 're 've 'm 'll 'd (含大写, 忽略大小写)
    if (c0 == '\'' && p[c0_len] != '\0') {
        static const char* sufs[] = {"s","t","re","ve","m","ll","d"};
        for (int i = 0; i < 7; i++) {
            size_t slen = strlen(sufs[i]);
            int match = 1;
            for (size_t j = 0; j < slen; j++) {
                unsigned char c = p[c0_len + j];
                char lc = (c >= 'A' && c <= 'Z') ? c + 32 : c;
                if (lc != sufs[i][j]) { match = 0; break; }
            }
            if (match) return c0_len + (int)slen;
        }
    }

    int cls0 = classify_char(c0);

    // 分支2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    //   可选前导: 单个"非字母非数字非换行"字符(含空格/标点), 后跟>=1个字母
    if (cls0 != CL_LETTER && cls0 != CL_NUMBER && cls0 != CL_NEWLINE) {
        int q = c0_len;
        int c1_len;
        int c1 = utf8_to_unicode((const char*)(p + q), &c1_len);
        if (c1 != 0 && classify_char(c1) == CL_LETTER) {
            // 前导 + 字母串
            q += c1_len;
            while (p[q]) {
                int cl;
                int c = utf8_to_unicode((const char*)(p + q), &cl);
                if (classify_char(c) != CL_LETTER) break;
                q += cl;
            }
            return q;
        }
        // 不是字母词开头, 落到分支4继续
    }

    // 分支2b: 纯字母词开头: [\p{L}\p{M}]+
    if (cls0 == CL_LETTER) {
        int q = c0_len;
        while (p[q]) {
            int cl;
            int c = utf8_to_unicode((const char*)(p + q), &cl);
            if (classify_char(c) != CL_LETTER) break;
            q += cl;
        }
        return q;
    }

    // 分支3: 单个数字
    if (cls0 == CL_NUMBER) return c0_len;

    // 分支4:  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*   (可选单个字面空格' ' + 标点团 + 可选换行)
    // 注意正则里 " ?" 是字面空格 U+0020, '\t' 等其他空白不算前缀
    // 注意: 标点团扩展时遇到特殊 token 起始必须截断 (HF 先抽特殊 token 再跑正则)
    if (cls0 == CL_OTHER || (cls0 == CL_SPACE && c0 == ' ')) {
        int q = (cls0 == CL_SPACE) ? c0_len : 0;
        // 标点团: [^\s\p{L}\p{M}\p{N}]+ (仅 OTHER)
        int c1_len;
        int c1 = utf8_to_unicode((const char*)(p + q), &c1_len);
        if (c1 != 0 && classify_char(c1) == CL_OTHER) {
            while (p[q]) {
                // 特殊 token 边界: 停止扩展, 留给下一段
                int sp_len;
                if (is_special_token(tok, (const char*)(p + q), &sp_id, &sp_len)) break;
                int cl;
                int c = utf8_to_unicode((const char*)(p + q), &cl);
                if (classify_char(c) != CL_OTHER) break;
                q += cl;
            }
            // 可选 [\r\n]*  (同样不能越过特殊 token)
            while (p[q]) {
                int sp_len;
                if (is_special_token(tok, (const char*)(p + q), &sp_id, &sp_len)) break;
                int cl;
                int c = utf8_to_unicode((const char*)(p + q), &cl);
                if (c != '\r' && c != '\n') break;
                q += cl;
            }
            return q;
        }
        // 空格后没接标点团 → 落到空白分支 (cls0==OTHER 时必然已返回)
    }

    // 分支5/6/7: 空白处理 (cls0 ∈ {SPACE, NEWLINE})
    if (cls0 == CL_SPACE || cls0 == CL_NEWLINE) {
        int q, last_nl_end;
        scan_space_run(p, c0_len, &q, &last_nl_end);

        // 关键: 若空白段后紧跟特殊 token, 则本片段以"段边界"结束:
        //   - 有换行 → 整段取到段尾 (HF: \s*[\r\n]+ 贪婪 + special 隔断, 实测 ' \n<|x|>' → [' \n', special])
        //   - 无换行 → 整段空白归分支6 \s+(?!\S) (特殊 token 前无 \S), '  <|x|>' → ['ĠĠ', special]
        int sp_id2, sp_len2;
        if (is_special_token(tok, (const char*)(p + q), &sp_id2, &sp_len2)) {
            return q;
        }
        // 分支5: \s*[\r\n]+ — 含换行的空白段, 匹配到"最后一个换行团"结束
        if (last_nl_end >= 0) return last_nl_end;
        // 分支6: \s+(?!\S) — 纯空格段: 后面还有非空白则让出最后一个空格
        if (p[q] != '\0') {
            if (q >= 2) return q - 1;   // 让出最后一个空格, 它会作为下一段的前缀
            return q;                    // 段长==1: \s+(?!\S) 失败, 由分支7 \s+ 匹配整段
        }
        return q;  // 文末空白: 整段归分支6
    }

    // 兜底: 单字符
    return c0_len;
}

// ==================== end pre_tokenizer ====================

// 编码主函数
int tokenizer_encode(BPETokenizer* tok, const char* text, int* token_ids, int max_tokens) {
    int num_tokens = 0;
    
    // NFC 归一化 (与 HF normalizer: NFC 对齐)
    int text_len = (int)strlen(text);
    char* nfc_buf = (char*)malloc(text_len * 4 + 16);
    if (!nfc_buf) return 0;
    nfc_normalize(text, nfc_buf, text_len * 4 + 16);
    const char* p = nfc_buf;
    
    while (*p && num_tokens < max_tokens) {
        int is_special = 0, special_id = -1;
        int seg_len = next_pretoken(tok, p, &is_special, &special_id);
        if (seg_len <= 0) break;
        
        if (is_special) {
            token_ids[num_tokens++] = special_id;
            p += seg_len;
            continue;
        }
        
        // 该片段的字节流做 BPE (片段可任意长, bpe_encode_bytes 动态分配)
        int n = bpe_encode_bytes(tok, (const unsigned char*)p, seg_len,
                                 token_ids + num_tokens, max_tokens - num_tokens);
        num_tokens += n;
        p += seg_len;
    }
    
    free(nfc_buf);
    return num_tokens;
}

// 解码主函数
int tokenizer_decode(BPETokenizer* tok, const int* token_ids, int num_tokens, char* output, int max_output) {
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    
    // 动态缓冲: 累计所有 token 的原始字节, 无固定 256 上限
    int byte_cap = num_tokens * MAX_TOKEN_LEN + 16;
    unsigned char* byte_buffer = (unsigned char*)malloc(byte_cap);
    if (!byte_buffer) return 0;
    int byte_len = 0;
    
    for (int i = 0; i < num_tokens; i++) {
        int id = token_ids[i];
        if (id < 0 || id >= tok->vocab_size) {
            continue;
        }
        
        const char* token = tok->vocab[id];
        if (!token) {
            continue;
        }
        
        // 将 GPT-2 字符转换回原始字节
        const char* p = token;
        while (*p) {
            int char_len;
            int unicode = utf8_to_unicode(p, &char_len);
            
            // 使用反向映射表查找原始字节
            int byte = internal->unicode_to_byte[unicode];
            
            if (byte >= 0) {
                if (byte_len < byte_cap) byte_buffer[byte_len++] = (unsigned char)byte;
            } else {
                // 不在映射表中，直接复制字节
                for (int j = 0; j < char_len; j++) {
                    if (byte_len < byte_cap) byte_buffer[byte_len++] = (unsigned char)p[j];
                }
            }
            p += char_len;
        }
    }
    
    // 将字节转换为 UTF-8 输出
    int out_len = 0;
    for (int i = 0; i < byte_len && out_len < max_output - 1; i++) {
        output[out_len++] = byte_buffer[i];
    }
    output[out_len] = '\0';
    
    free(byte_buffer);
    return out_len;
}

// 获取 token 字符串
const char* tokenizer_get_token(BPETokenizer* tok, int id) {
    if (id >= 0 && id < tok->vocab_size) {
        return tok->vocab[id];
    }
    return NULL;
}

// 获取 token ID
int tokenizer_get_id(BPETokenizer* tok, const char* token) {
    TokenizerInternal* internal = (TokenizerInternal*)tok->cache;
    int found = 0;
    return hashmap_get(&internal->token_to_id, token, &found);
}

// 测试函数
void test_tokenizer(const char* vocab_path, const char* merges_path) {
    BPETokenizer tok;
    
    printf("\n========== BPE Tokenizer 测试 ==========\n\n");
    
    // 初始化
    if (tokenizer_init(&tok) != 0) {
        fprintf(stderr, "初始化失败\n");
        return;
    }
    
    // 加载词表
    if (tokenizer_load_vocab(&tok, vocab_path) != 0) {
        tokenizer_free(&tok);
        return;
    }
    
    // 加载合并规则
    if (tokenizer_load_merges(&tok, merges_path) != 0) {
        tokenizer_free(&tok);
        return;
    }
    
    // 加载特殊 token
    tokenizer_load_special_tokens(&tok);
    
    printf("\n---------- 测试编码解码 ----------\n\n");
    
    // 测试用例
    struct { const char* text; const char* desc; } test_cases[] = {
        {"Hello", "简单英文"},
        {"Hello world", "带空格英文"},
        {"你好", "简单中文"},
        {"你好世界", "中文词组"},
        {"Hello 世界", "中英混合"},
        {"The quick brown fox jumps over the lazy dog.", "英文句子"},
        {"人工智能正在改变世界", "中文句子"},
        {"<|im_start|>user\nHello<|im_end|>", "特殊token"},
        {"1 + 2 = 3", "数字和符号"},
        {"测试 Test 123", "混合测试"},
        {NULL, NULL}
    };
    
    int token_ids[256];
    char decoded[512];
    
    for (int i = 0; test_cases[i].text; i++) {
        printf("测试 %d: %s\n", i + 1, test_cases[i].desc);
        printf("原文: \"%s\"\n", test_cases[i].text);
        
        // 编码
        int num_tokens = tokenizer_encode(&tok, test_cases[i].text, token_ids, 256);
        printf("Token 数量: %d\n", num_tokens);
        
        printf("Token IDs: ");
        for (int j = 0; j < num_tokens && j < 20; j++) {
            printf("%d ", token_ids[j]);
        }
        if (num_tokens > 20) printf("...");
        printf("\n");
        
        // 显示对应的 token 字符串
        printf("Tokens: ");
        for (int j = 0; j < num_tokens && j < 10; j++) {
            const char* token = tokenizer_get_token(&tok, token_ids[j]);
            if (token) {
                printf("[%s] ", token);
            }
        }
        if (num_tokens > 10) printf("...");
        printf("\n");
        
        // 解码
        int decoded_len = tokenizer_decode(&tok, token_ids, num_tokens, decoded, sizeof(decoded));
        printf("解码结果: \"%s\"\n", decoded);
        
        // 验证
        if (strcmp(decoded, test_cases[i].text) == 0) {
            printf("✓ 编解码一致\n");
        } else {
            printf("✗ 编解码不一致\n");
            printf("  原文长度: %zu, 解码长度: %d\n", strlen(test_cases[i].text), decoded_len);
        }
        
        printf("\n");
    }
    
    // 内存使用报告
    printf("---------- 内存使用 ----------\n");
    printf("词表大小: %d tokens\n", tok.vocab_size);
    printf("合并规则: %d 条\n", tok.num_merges);
    printf("特殊 token: %d 个\n", tok.num_special_tokens);
    
    size_t vocab_mem = tok.vocab_size * sizeof(char*) + 
                       tok.vocab_size * 10;  // 平均每个 token 10 字节
    size_t merge_mem = tok.num_merges * sizeof(*tok.merges) +
                       tok.num_merges * 30;  // 平均每条规则 30 字节
    size_t hash_mem = HASH_SIZE * sizeof(HashEntry*) * 2;  // 两个哈希表
    
    printf("估算内存使用:\n");
    printf("  词表: ~%.2f MB\n", vocab_mem / 1024.0 / 1024.0);
    printf("  合并规则: ~%.2f MB\n", merge_mem / 1024.0 / 1024.0);
    printf("  哈希表: ~%.2f MB\n", hash_mem / 1024.0 / 1024.0);
    printf("  总计: ~%.2f MB\n", (vocab_mem + merge_mem + hash_mem) / 1024.0 / 1024.0);
    
    printf("\n==========================================\n");
    
    tokenizer_free(&tok);
}

// 主函数（用于独立测试）
#ifdef TOKENIZER_MAIN
int main(int argc, char** argv) {
    const char* vocab_path = "/mnt/Qwen3.5-0.8B/vocab.json";
    const char* merges_path = "/mnt/Qwen3.5-0.8B/merges.txt";
    
    if (argc >= 3) {
        vocab_path = argv[1];
        merges_path = argv[2];
    }
    
    test_tokenizer(vocab_path, merges_path);
    return 0;
}
#endif