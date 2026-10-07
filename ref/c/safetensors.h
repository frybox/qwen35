#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

// Data type enumeration
typedef enum {
    DTYPE_F32,      // float32
    DTYPE_F16,      // float16
    DTYPE_BF16,     // bfloat16
    DTYPE_I32,      // int32
    DTYPE_I16,      // int16
    DTYPE_I8,       // int8
    DTYPE_U8,       // uint8
    DTYPE_UNKNOWN
} DType;

// Tensor metadata
typedef struct {
    char *name;             // tensor name
    DType dtype;            // data type
    int ndim;               // number of dimensions
    int64_t shape[8];       // shape (max 8 dimensions)
    int64_t data_offset;    // offset in data section
    int64_t data_size;      // size in bytes
} TensorMeta;

// Safetensors file handle
typedef struct {
    char *filepath;         // file path
    uint64_t metadata_len;  // JSON metadata length
    char *metadata;         // JSON metadata string
    int num_tensors;        // number of tensors
    TensorMeta *tensors;    // tensor metadata array
    int64_t data_start;     // data section start offset (8 + metadata_len)
    FILE *fp;               // file pointer (NULL after close)
} SafetensorsFile;

// Parse dtype string to enum
DType parse_dtype(const char *str);

// Get size in bytes for a dtype
size_t dtype_size(DType dtype);

// Open and parse safetensors file
// Returns NULL on error
SafetensorsFile* safetensors_open(const char *filepath);

// Close and free safetensors file
void safetensors_close(SafetensorsFile *sf);

// Get tensor metadata by name
// Returns NULL if not found
TensorMeta* safetensors_get_tensor(SafetensorsFile *sf, const char *name);

// Load tensor data into buffer
// Buffer must be at least tensor->data_size bytes
// Returns 0 on success, -1 on error
int safetensors_load_tensor(SafetensorsFile *sf, TensorMeta *tensor, void *buffer);

// Load tensor data as float32 (converts BF16/F16 if needed)
// Buffer must be at least tensor->data_size * sizeof(float) bytes (for BF16/F16)
// Returns 0 on success, -1 on error
int safetensors_load_tensor_as_f32(SafetensorsFile *sf, TensorMeta *tensor, float *buffer);

// Print all tensor names and shapes
void safetensors_print_info(SafetensorsFile *sf);

// Test function - verify file can be parsed and print summary
int safetensors_test(const char *filepath);

#endif // SAFETENSORS_H