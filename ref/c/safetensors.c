#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Parse dtype string to enum
DType parse_dtype(const char *str) {
    if (strcmp(str, "F32") == 0) return DTYPE_F32;
    if (strcmp(str, "F16") == 0) return DTYPE_F16;
    if (strcmp(str, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(str, "I32") == 0) return DTYPE_I32;
    if (strcmp(str, "I16") == 0) return DTYPE_I16;
    if (strcmp(str, "I8") == 0) return DTYPE_I8;
    if (strcmp(str, "U8") == 0) return DTYPE_U8;
    return DTYPE_UNKNOWN;
}

// Get size in bytes for a dtype
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_F32: return 4;
        case DTYPE_F16: return 2;
        case DTYPE_BF16: return 2;
        case DTYPE_I32: return 4;
        case DTYPE_I16: return 2;
        case DTYPE_I8: return 1;
        case DTYPE_U8: return 1;
        default: return 0;
    }
}

// Simple JSON parsing helpers (no external dependencies)
static char* json_find_key(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    char *pos = strstr(json, search);
    if (!pos) return NULL;
    pos += strlen(search);
    while (*pos && isspace(*pos)) pos++;
    if (*pos == ':') pos++;
    while (*pos && isspace(*pos)) pos++;
    return pos;
}

static int json_parse_string(const char *json, char *buf, size_t buflen) {
    if (*json != '"') return -1;
    json++;
    size_t i = 0;
    while (*json && *json != '"' && i < buflen - 1) {
        buf[i++] = *json++;
    }
    buf[i] = '\0';
    return 0;
}

static int json_parse_int(const char *json, int64_t *val) {
    *val = 0;
    int neg = 0;
    if (*json == '-') {
        neg = 1;
        json++;
    }
    while (*json && isdigit(*json)) {
        *val = *val * 10 + (*json - '0');
        json++;
    }
    if (neg) *val = -*val;
    return 0;
}

// Parse tensor metadata from JSON
// key_start: points to opening quote of tensor name
// value_start: points to '{' of tensor value object
static int parse_tensor_meta(const char *key_start,
                             const char *value_start, 
                             TensorMeta *meta, int64_t data_base) {
    // Extract name from key (between quotes)
    const char *name_start = key_start + 1; // skip opening quote
    const char *name_end = name_start;
    while (*name_end && *name_end != '"') name_end++;
    
    size_t name_len = name_end - name_start;
    meta->name = malloc(name_len + 1);
    memcpy(meta->name, name_start, name_len);
    meta->name[name_len] = '\0';
    
    // Parse dtype
    const char *dtype_pos = json_find_key(value_start, "dtype");
    if (dtype_pos) {
        char dtype_str[32];
        json_parse_string(dtype_pos, dtype_str, sizeof(dtype_str));
        meta->dtype = parse_dtype(dtype_str);
    } else {
        meta->dtype = DTYPE_UNKNOWN;
    }
    
    // Parse shape
    const char *shape_pos = json_find_key(value_start, "shape");
    meta->ndim = 0;
    if (shape_pos && *shape_pos == '[') {
        shape_pos++;
        while (*shape_pos && *shape_pos != ']') {
            if (isdigit(*shape_pos)) {
                json_parse_int(shape_pos, &meta->shape[meta->ndim]);
                meta->ndim++;
                while (*shape_pos && *shape_pos != ',' && *shape_pos != ']') shape_pos++;
            }
            if (*shape_pos == ',') shape_pos++;
        }
    }
    
    // Parse data_offsets
    const char *offset_pos = json_find_key(value_start, "data_offsets");
    int64_t offset_start = 0, offset_end = 0;
    if (offset_pos && *offset_pos == '[') {
        offset_pos++;
        const char *first = offset_pos;
        while (*first && !isdigit(*first)) first++;
        if (*first) json_parse_int(first, &offset_start);
        
        const char *second = strstr(first, ",");
        if (second) {
            second++;
            while (*second && !isdigit(*second)) second++;
            if (*second) json_parse_int(second, &offset_end);
        }
    }
    
    meta->data_offset = data_base + offset_start;
    meta->data_size = offset_end - offset_start;
    
    return 0;
}

// Open and parse safetensors file
SafetensorsFile* safetensors_open(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s\n", filepath);
        return NULL;
    }
    
    // Read metadata length (8 bytes little-endian)
    uint64_t metadata_len;
    if (fread(&metadata_len, 8, 1, fp) != 1) {
        fprintf(stderr, "Error: cannot read metadata length\n");
        fclose(fp);
        return NULL;
    }
    
    // Read metadata
    char *metadata = malloc(metadata_len + 1);
    if (fread(metadata, 1, metadata_len, fp) != metadata_len) {
        fprintf(stderr, "Error: cannot read metadata\n");
        free(metadata);
        fclose(fp);
        return NULL;
    }
    metadata[metadata_len] = '\0';
    
    SafetensorsFile *sf = calloc(1, sizeof(SafetensorsFile));
    sf->filepath = strdup(filepath);
    sf->metadata_len = metadata_len;
    sf->metadata = metadata;
    sf->data_start = 8 + metadata_len;
    sf->fp = fp;
    
    // Count and parse tensors
    // We need to find tensor entries (those with dtype, shape, data_offsets)
    int num_tensors = 0;
    int capacity = 256;
    sf->tensors = malloc(capacity * sizeof(TensorMeta));
    
    // Parse JSON manually - find all tensor entries
    // Format: "tensor_name":{"dtype":"...","shape":[...],"data_offsets":[...]}
    const char *p = metadata;
    while (*p) {
        // Look for pattern: "name":{
        if (*p == '"' && *(p+1) != '_' && strncmp(p+1, "__metadata__", 12) != 0) {
            // Find the closing quote
            const char *key_start = p;
            const char *key_end = p + 1;
            while (*key_end && *key_end != '"') key_end++;
            
            // Check if next non-whitespace is ':'
            const char *after_key = key_end + 1;
            while (*after_key && isspace(*after_key)) after_key++;
            
            if (*after_key == ':') {
                after_key++;
                while (*after_key && isspace(*after_key)) after_key++;
                
                // Check if this is an object (starts with '{')
                if (*after_key == '{') {
                    // Check if this has "dtype" field (is a tensor)
                    const char *dtype_pos = json_find_key(after_key, "dtype");
                    if (dtype_pos) {
                        if (num_tensors >= capacity) {
                            capacity *= 2;
                            sf->tensors = realloc(sf->tensors, capacity * sizeof(TensorMeta));
                        }
                        
                        if (parse_tensor_meta(key_start, after_key, 
                                             &sf->tensors[num_tensors], sf->data_start) == 0) {
                            num_tensors++;
                        }
                    }
                }
            }
            p = key_end;
        }
        p++;
    }
    
    sf->num_tensors = num_tensors;
    return sf;
}

// Close and free safetensors file
void safetensors_close(SafetensorsFile *sf) {
    if (!sf) return;
    
    if (sf->fp) fclose(sf->fp);
    free(sf->filepath);
    free(sf->metadata);
    
    for (int i = 0; i < sf->num_tensors; i++) {
        free(sf->tensors[i].name);
    }
    free(sf->tensors);
    free(sf);
}

// Get tensor metadata by name
TensorMeta* safetensors_get_tensor(SafetensorsFile *sf, const char *name) {
    if (!sf) return NULL;
    
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) {
            return &sf->tensors[i];
        }
    }
    return NULL;
}

// Load tensor data into buffer
int safetensors_load_tensor(SafetensorsFile *sf, TensorMeta *tensor, void *buffer) {
    if (!sf || !tensor || !buffer || !sf->fp) return -1;
    
    fseek(sf->fp, tensor->data_offset, SEEK_SET);
    if (fread(buffer, 1, (size_t)tensor->data_size, sf->fp) != (size_t)tensor->data_size) {
        return -1;
    }
    
    return 0;
}

// Convert BF16 to F32
static void bf16_to_f32(const uint16_t *bf16, float *f32, size_t count) {
    for (size_t i = 0; i < count; i++) {
        // BF16 is just the upper 16 bits of an F32
        uint32_t val = ((uint32_t)bf16[i]) << 16;
        memcpy(&f32[i], &val, sizeof(float));
    }
}

// Convert F16 to F32 (simple implementation)
static void f16_to_f32(const uint16_t *f16, float *f32, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint16_t h = f16[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        
        if (exp == 0) {
            if (mant == 0) {
                // Zero
                uint32_t f = sign << 31;
                memcpy(&f32[i], &f, sizeof(float));
            } else {
                // Denormal
                exp = 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
                uint32_t f = (sign << 31) | ((127 - 15 + exp) << 23) | (mant << 13);
                memcpy(&f32[i], &f, sizeof(float));
            }
        } else if (exp == 31) {
            // Inf or NaN
            uint32_t f = (sign << 31) | 0x7F800000 | (mant << 13);
            memcpy(&f32[i], &f, sizeof(float));
        } else {
            // Normal
            uint32_t f = (sign << 31) | ((127 - 15 + exp) << 23) | (mant << 13);
            memcpy(&f32[i], &f, sizeof(float));
        }
    }
}

// Load tensor data as float32
int safetensors_load_tensor_as_f32(SafetensorsFile *sf, TensorMeta *tensor, float *buffer) {
    if (!sf || !tensor || !buffer || !sf->fp) return -1;
    
    size_t elem_count = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        elem_count *= tensor->shape[i];
    }
    
    if (tensor->dtype == DTYPE_F32) {
        return safetensors_load_tensor(sf, tensor, buffer);
    } else if (tensor->dtype == DTYPE_BF16) {
        uint16_t *tmp = malloc(tensor->data_size);
        if (safetensors_load_tensor(sf, tensor, tmp) != 0) {
            free(tmp);
            return -1;
        }
        bf16_to_f32(tmp, buffer, elem_count);
        free(tmp);
        return 0;
    } else if (tensor->dtype == DTYPE_F16) {
        uint16_t *tmp = malloc(tensor->data_size);
        if (safetensors_load_tensor(sf, tensor, tmp) != 0) {
            free(tmp);
            return -1;
        }
        f16_to_f32(tmp, buffer, elem_count);
        free(tmp);
        return 0;
    }
    
    return -1; // unsupported dtype
}

// Print all tensor names and shapes
void safetensors_print_info(SafetensorsFile *sf) {
    if (!sf) return;
    
    printf("Safetensors file: %s\n", sf->filepath);
    printf("Metadata length: %lu bytes\n", (unsigned long)sf->metadata_len);
    printf("Number of tensors: %d\n\n", sf->num_tensors);
    
    printf("%-80s %-8s %s\n", "Name", "Dtype", "Shape");
    printf("%-80s %-8s %s\n", "----", "-----", "-----");
    
    for (int i = 0; i < sf->num_tensors; i++) {
        TensorMeta *t = &sf->tensors[i];
        printf("%-80s ", t->name);
        
        const char *dtype_str = "UNKNOWN";
        switch (t->dtype) {
            case DTYPE_F32: dtype_str = "F32"; break;
            case DTYPE_F16: dtype_str = "F16"; break;
            case DTYPE_BF16: dtype_str = "BF16"; break;
            case DTYPE_I32: dtype_str = "I32"; break;
            default: break;
        }
        printf("%-8s ", dtype_str);
        
        printf("[");
        for (int j = 0; j < t->ndim; j++) {
            printf("%ld", (long)t->shape[j]);
            if (j < t->ndim - 1) printf(", ");
        }
        printf("]\n");
    }
}

// Test function
int safetensors_test(const char *filepath) {
    printf("Testing safetensors parser with: %s\n\n", filepath);
    
    SafetensorsFile *sf = safetensors_open(filepath);
    if (!sf) {
        fprintf(stderr, "Failed to open safetensors file\n");
        return -1;
    }
    
    safetensors_print_info(sf);
    
    printf("\n--- Testing tensor loading ---\n");
    
    // Try loading the first tensor
    if (sf->num_tensors > 0) {
        TensorMeta *t = &sf->tensors[0];
        printf("\nLoading tensor: %s\n", t->name);
        printf("  Dtype: %d, Size: %ld bytes\n", t->dtype, (long)t->data_size);
        
        size_t elem_count = 1;
        for (int i = 0; i < t->ndim; i++) {
            elem_count *= t->shape[i];
        }
        printf("  Elements: %zu\n", elem_count);
        
        float *data = malloc(elem_count * sizeof(float));
        if (safetensors_load_tensor_as_f32(sf, t, data) == 0) {
            printf("  First 5 values: ");
            for (size_t i = 0; i < 5 && i < elem_count; i++) {
                printf("%.6f ", data[i]);
            }
            printf("\n");
            printf("  SUCCESS: Tensor loaded correctly\n");
        } else {
            printf("  ERROR: Failed to load tensor\n");
        }
        free(data);
    }
    
    safetensors_close(sf);
    printf("\nTest completed.\n");
    return 0;
}

// Main function for standalone testing
#ifdef SAFETENSORS_MAIN
int main(int argc, char **argv) {
    const char *filepath = "/mnt/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
    if (argc > 1) {
        filepath = argv[1];
    }
    return safetensors_test(filepath);
}
#endif