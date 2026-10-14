/*
 * Qwen3.5-0.8B C Inference Engine
 * Direct safetensors loading implementation
 * 
 * Architecture: 24 layers, mixed attention (18x GDN + 6x full attention)
 * - Layers 0-2, 4-6, 8-10, 12-14, 16-18, 20-22: linear_attn (GDN)
 * - Layers 3, 7, 11, 15, 19, 23: self_attn (full attention)
 *
 * Usage:
 *   ./qwen35_engine "你的提示词"      # prompt from the command line
 *   ./qwen35_engine                   # built-in demo prompt
 * Env overrides:
 *   QWEN35_PROMPT       prompt text (used when no argv[1] is given)
 *   QWEN35_PROMPT_IDS   comma separated token ids, bypasses the tokenizer
 *   QWEN35_MODEL        safetensors path
 *   QWEN35_VOCAB        vocab.json path (merges.txt is looked up next to it)
 *   QWEN35_DUMP_LOGITS  dump the last-position logits to a binary file
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Monotonic wall-clock timer, used for the prefill / decode throughput stats.
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#include "safetensors.h"
#include "tokenizer.h"

// ============================================================================
// Config
// ============================================================================

typedef struct {
    int dim;           // 1024
    int ffndim;        // 3584
    int nlayers;       // 24
    int nheads;        // 8
    int nkvheads;      // 2
    int head_dim;      // 256
    int nvocab;        // 248320
    int nctx;          // 4096 (context length)
    int gdn_vheads;    // 16
    int gdn_kdim;      // 128
    int gdn_vdim;      // 128
    int layer_types[24];  // 0=linear_attn, 1=full_attn
} Config;

// ============================================================================
// Weights - dynamically allocated float arrays
// ============================================================================

typedef struct {
    float *embed_tokens;    // [nvocab, dim]
    float *embed_norm;      // [dim] - final norm
    
    // Per-layer weights (indexed by layer)
    float *ln1[24];         // input_layernorm [dim]
    float *ln2[24];         // post_attention_layernorm [dim]
    
    // GDN/linear_attn weights (used when layer_types[l] == 0)
    float *gdn_qkv[24];     // in_proj_qkv [6144, dim]
    float *gdn_z[24];       // in_proj_z [2048, dim]
    float *gdn_a[24];       // in_proj_a [16, dim] -> [16*dim]
    float *gdn_b[24];       // in_proj_b [16, dim] -> [16*dim]
    float *gdn_A[24];       // A_log [16] (stored as log, need exp)
    float *gdn_dt[24];      // dt_bias [16]
    float *gdn_conv[24];    // conv1d.weight [6144, 1, 4] -> [6144, 4]
    float *gdn_norm[24];    // norm.weight [128]
    float *gdn_out[24];     // out_proj [1024, 2048]
    
    // Full attention weights (used when layer_types[l] == 1)
    float *fa_q[24];        // q_proj [4096, dim] (includes gate)
    float *fa_k[24];        // k_proj [512, dim]
    float *fa_v[24];        // v_proj [512, dim]
    float *fa_o[24];        // o_proj [1024, 2048]
    float *fa_qn[24];       // q_norm [256]
    float *fa_kn[24];       // k_norm [256]
    
    // MLP weights (all layers)
    float *gate[24];        // gate_proj [3584, dim]
    float *up[24];          // up_proj [3584, dim]
    float *down[24];        // down_proj [1024, 3584]
} Weights;

// ============================================================================
// State
// ============================================================================

typedef struct {
    float *x, *x1, *x2;     // [dim] working buffers
    float *h, *h1;          // [ffndim] working buffers
    float *logits;          // [nvocab] output logits
    
    // Full attention KV cache
    float *kcache, *vcache; // [nlayers, nctx, kvdim]
    float *q, *attn;        // [nheads*head_dim], [nheads*nctx]
    
    // GDN state
    float *gdn_state;       // [ngdn_layers, vheads, kdim, vdim]
    float *conv_state;      // [ngdn_layers, 6144, 4]
    float *gdn_out;         // [vheads, vdim] = [16, 128] = 2048
    float *gate_buf;        // [nheads * head_dim] q_proj gate for full attention
} State;

typedef struct {
    Config c;
    Weights w;
    State s;
    SafetensorsFile *sf;
} Transformer;

// ============================================================================
// Utils
// ============================================================================

static void die(const char *msg) { 
    fprintf(stderr, "Error: %s\n", msg); 
    exit(1); 
}

static void check_ptr(void *ptr, const char *msg) { 
    if (!ptr) die(msg); 
}

void *xmalloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) die("malloc failed");
    return p;
}

// ============================================================================
// Config initialization
// ============================================================================

void init_config(Config *c) {
    c->dim = 1024;
    c->ffndim = 3584;
    c->nlayers = 24;
    c->nheads = 8;
    c->nkvheads = 2;
    c->head_dim = 256;
    c->nvocab = 248320;
    c->nctx = 4096;
    c->gdn_vheads = 16;
    c->gdn_kdim = 128;
    c->gdn_vdim = 128;
    
    // Layer types: 0-2=linear, 3=full, 4-6=linear, 7=full, ...
    // Full attention at layers 3, 7, 11, 15, 19, 23
    for (int i = 0; i < 24; i++) {
        c->layer_types[i] = (i % 4 == 3) ? 1 : 0;
    }
}

// ============================================================================
// Weight loading from safetensors
// ============================================================================

// Helper to load a tensor and convert to F32
float *load_tensor_f32(SafetensorsFile *sf, const char *name, int64_t expected_elems) {
    TensorMeta *meta = safetensors_get_tensor(sf, name);
    if (!meta) {
        fprintf(stderr, "Warning: tensor '%s' not found\n", name);
        return NULL;
    }
    
    int64_t elems = 1;
    for (int i = 0; i < meta->ndim; i++) {
        elems *= meta->shape[i];
    }
    
    if (expected_elems > 0 && elems != expected_elems) {
        fprintf(stderr, "Warning: tensor '%s' has %ld elements, expected %ld\n", 
                name, (long)elems, (long)expected_elems);
    }
    
    float *data = xmalloc(elems * sizeof(float));
    if (safetensors_load_tensor_as_f32(sf, meta, data) != 0) {
        fprintf(stderr, "Error: failed to load tensor '%s'\n", name);
        free(data);
        return NULL;
    }
    
    return data;
}

void load_weights(Transformer *t, const char *safetensors_path) {
    t->sf = safetensors_open(safetensors_path);
    if (!t->sf) die("failed to open safetensors file");
    
    Config *c = &t->c;
    char name[256];
    
    printf("Loading weights from safetensors...\n");
    
    // Embedding
    t->w.embed_tokens = load_tensor_f32(t->sf, 
        "model.language_model.embed_tokens.weight", c->nvocab * c->dim);
    check_ptr(t->w.embed_tokens, "failed to load embed_tokens");
    
    // Final norm (using embed_norm for final layer norm)
    // Note: The model might have a separate final_norm, but typically shares with last layer
    // For Qwen models, final norm is usually model.language_model.norm.weight
    // Let's check if it exists, otherwise we'll use the first layer's norm as placeholder
    TensorMeta *final_norm = safetensors_get_tensor(t->sf, "model.language_model.norm.weight");
    if (final_norm) {
        t->w.embed_norm = load_tensor_f32(t->sf, "model.language_model.norm.weight", c->dim);
    } else {
        // Use layer 0's ln as placeholder (will need to verify)
        t->w.embed_norm = xmalloc(c->dim * sizeof(float));
        memset(t->w.embed_norm, 0, c->dim * sizeof(float));
        for (int i = 0; i < c->dim; i++) t->w.embed_norm[i] = 1.0f; // default to ones
    }
    
    // Per-layer weights
    for (int l = 0; l < c->nlayers; l++) {
        // Layer norms
        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", l);
        t->w.ln1[l] = load_tensor_f32(t->sf, name, c->dim);
        check_ptr(t->w.ln1[l], "failed to load input_layernorm");
        
        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
        t->w.ln2[l] = load_tensor_f32(t->sf, name, c->dim);
        check_ptr(t->w.ln2[l], "failed to load post_attention_layernorm");
        
        if (c->layer_types[l] == 0) {
            // Linear attention (GDN) layer
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", l);
            t->w.gdn_qkv[l] = load_tensor_f32(t->sf, name, 6144 * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", l);
            t->w.gdn_z[l] = load_tensor_f32(t->sf, name, 2048 * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
            t->w.gdn_a[l] = load_tensor_f32(t->sf, name, c->gdn_vheads * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
            t->w.gdn_b[l] = load_tensor_f32(t->sf, name, c->gdn_vheads * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", l);
            t->w.gdn_A[l] = load_tensor_f32(t->sf, name, c->gdn_vheads);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", l);
            t->w.gdn_dt[l] = load_tensor_f32(t->sf, name, c->gdn_vheads);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", l);
            t->w.gdn_conv[l] = load_tensor_f32(t->sf, name, 6144 * 4);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", l);
            t->w.gdn_norm[l] = load_tensor_f32(t->sf, name, c->gdn_kdim);  // [128]
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", l);
            t->w.gdn_out[l] = load_tensor_f32(t->sf, name, c->dim * (c->gdn_vheads * c->gdn_vdim));
            
        } else {
            // Full attention layer
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
            t->w.fa_q[l] = load_tensor_f32(t->sf, name, 4096 * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
            int kvdim = c->head_dim * c->nkvheads;  // 512
            t->w.fa_k[l] = load_tensor_f32(t->sf, name, kvdim * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
            t->w.fa_v[l] = load_tensor_f32(t->sf, name, kvdim * c->dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
            t->w.fa_o[l] = load_tensor_f32(t->sf, name, c->dim * (c->nheads * c->head_dim));
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", l);
            t->w.fa_qn[l] = load_tensor_f32(t->sf, name, c->head_dim);
            
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", l);
            t->w.fa_kn[l] = load_tensor_f32(t->sf, name, c->head_dim);
        }
        
        // MLP weights (all layers)
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", l);
        t->w.gate[l] = load_tensor_f32(t->sf, name, c->ffndim * c->dim);
        
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.up_proj.weight", l);
        t->w.up[l] = load_tensor_f32(t->sf, name, c->ffndim * c->dim);
        
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.down_proj.weight", l);
        t->w.down[l] = load_tensor_f32(t->sf, name, c->dim * c->ffndim);
    }
    
    printf("Weights loaded successfully\n");
}

// ============================================================================
// State allocation
// ============================================================================

void malloc_state(State *s, Config *c) {
    int head_dim = c->head_dim;  // 256 (from config, not computed)
    int kvdim = head_dim * c->nkvheads; // 512
    int ngdn = 0;
    for (int i = 0; i < c->nlayers; i++) {
        if (c->layer_types[i] == 0) ngdn++;
    }
    
    s->x = xmalloc(c->dim * sizeof(float));
    s->x1 = xmalloc(c->dim * sizeof(float));
    s->x2 = xmalloc(c->dim * sizeof(float));
    s->h = xmalloc(6144 * sizeof(float));  // GDN QKV needs 6144 (larger than ffndim)
    s->h1 = xmalloc(4096 * sizeof(float));  // Full attention Q+gate needs 4096 (larger than ffndim)
    s->logits = xmalloc(c->nvocab * sizeof(float));
    
    // GDN output buffer (needs nv * vd = 16 * 128 = 2048)
    s->gdn_out = xmalloc(c->gdn_vheads * c->gdn_vdim * sizeof(float));
    
    // Full attention cache
    s->kcache = xmalloc(c->nlayers * c->nctx * kvdim * sizeof(float));
    s->vcache = xmalloc(c->nlayers * c->nctx * kvdim * sizeof(float));
    s->q = xmalloc(c->nheads * head_dim * sizeof(float));
    s->attn = xmalloc(c->nheads * c->nctx * sizeof(float));
    s->gate_buf = xmalloc(c->nheads * head_dim * sizeof(float));
    
    // GDN state
    s->gdn_state = xmalloc(ngdn * c->gdn_vheads * c->gdn_kdim * c->gdn_vdim * sizeof(float));
    s->conv_state = xmalloc(ngdn * 6144 * 4 * sizeof(float));
}

// ============================================================================
// Math operations
// ============================================================================

// Qwen3_5RMSNorm: output = (x * rsqrt(mean(x^2)+eps)) * (1 + weight)
// NOTE: the stored parameter is the *delta* around 0 (nn.Parameter(torch.zeros(dim))),
// the effective scale is (1 + weight). Used by input_layernorm, post_attention_layernorm,
// the final norm, and q_norm / k_norm.
void rmsnorm(float *o, float *x, float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = (1.0f + w[i]) * ss * x[i];
}

// softplus with the PyTorch default threshold=20 (F.softplus): linear above 20.
float softplus_f(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// Qwen3_5RMSNormGated: weight is a plain scale (nn.Parameter(torch.ones(dim))), NO 1+.
void rmsnorm_plain(float *o, float *x, float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = w[i] * ss * x[i];
}

void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void matmul(float *o, float *W, float *x, int n, int d) {
    // W: [d, n], x: [n], o: [d]
    for (int i = 0; i < d; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) sum += W[i*n + j] * x[j];
        o[i] = sum;
    }
}

float silu(float x) { 
    return x / (1.0f + expf(-x)); 
}

// ============================================================================
// RoPE (Rotary Position Embeddings)
// ============================================================================

void apply_rope(float *q, float *k, int head_dim, int nq, int nkv, int pos) {
    // Qwen3.5 text RoPE: partial_rotary_factor=0.25 -> rotary_dim = 64 of head_dim 256.
    // inv_freq[i] = 1 / theta^(2i/rotary_dim), i = 0..rotary_dim/2-1
    // Layout: cos/sin are the inv_freq pattern repeated twice, so the rotation pairs
    // dim i with dim i + rotary_dim/2 (GPT-NeoX / rotate_half style), NOT adjacent pairs.
    // Dims [rotary_dim, head_dim) pass through unchanged.
    const int rotary_dim = 64;
    const int half = rotary_dim / 2;  // 32
    const float theta = 10000000.0f;

    float cr[32], ci[32];
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)rotary_dim);
        float val = pos * freq;
        cr[i] = cosf(val);
        ci[i] = sinf(val);
    }

    for (int h = 0; h < nq; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float a = qh[i], b = qh[i + half];
            qh[i] = a * cr[i] - b * ci[i];
            qh[i + half] = b * cr[i] + a * ci[i];
        }
    }
    for (int h = 0; h < nkv; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float a = kh[i], b = kh[i + half];
            kh[i] = a * cr[i] - b * ci[i];
            kh[i + half] = b * cr[i] + a * ci[i];
        }
    }
}

// ============================================================================
// Full attention forward
// ============================================================================

void forward_full_attn(Transformer *t, int layer, int pos) {
    Config *c = &t->c;
    State *s = &t->s;
    Weights *w = &t->w;

    int dim = c->dim;
    int head_dim = c->head_dim;  // 256
    int kvdim = head_dim * c->nkvheads;  // 512
    int gqa = c->nheads / c->nkvheads;  // 4

    // Norm
    rmsnorm(s->x1, s->x, w->ln1[layer], dim);

    // Q projection: outputs [query, gate] each of size dim*4 = 4096
    // Shape: [4096, dim] -> output is 4096
    float *q_gate = s->h1;  // use h1 as temp buffer [dim*4] = 4096
    matmul(q_gate, w->fa_q[layer], s->x1, dim, dim * 4);

    // Reference: q_proj(hidden).view(..., nheads, head_dim*2) then torch.chunk(..., 2, dim=-1)
    // so per head h: Q = q_gate[h*2*hd : h*2*hd+hd], gate = q_gate[h*2*hd+hd : h*2*hd+2*hd].
    // NOT a flat split at offset 2048.
    int qsize = c->nheads * head_dim;  // 2048
    (void)qsize;
    float *q = s->q;
    float *gate = s->gate_buf;
    for (int h = 0; h < c->nheads; h++) {
        memcpy(q + h*head_dim, q_gate + h*2*head_dim, head_dim * sizeof(float));
        memcpy(gate + h*head_dim, q_gate + h*2*head_dim + head_dim, head_dim * sizeof(float));
    }

    // Q norm (per head) - q_norm has shape [256], applied BEFORE RoPE
    for (int h = 0; h < c->nheads; h++) {
        rmsnorm(q + h*head_dim, q + h*head_dim, w->fa_qn[layer], head_dim);
    }

    // K projection (kvdim=512), kept in s->h; V projection kept right after it.
    float *k = s->h;
    float *v = s->h + kvdim;
    matmul(k, w->fa_k[layer], s->x1, dim, kvdim);
    // K norm (per kv head), applied BEFORE RoPE
    for (int h = 0; h < c->nkvheads; h++) {
        rmsnorm(k + h*head_dim, k + h*head_dim, w->fa_kn[layer], head_dim);
    }
    matmul(v, w->fa_v[layer], s->x1, dim, kvdim);

    // RoPE
    apply_rope(q, k, head_dim, c->nheads, c->nkvheads, pos);

    // Cache KV
    size_t loff = (size_t)layer * c->nctx * kvdim;
    memcpy(s->kcache + loff + pos*kvdim, k, kvdim * sizeof(float));
    memcpy(s->vcache + loff + pos*kvdim, v, kvdim * sizeof(float));

    // Attention computation
    for (int h = 0; h < c->nheads; h++) {
        float *qh = q + h*head_dim;
        float *att = s->attn + h*c->nctx;
        int kvh = h / gqa;

        for (int t = 0; t <= pos; t++) {
            float *kt = s->kcache + loff + t*kvdim + kvh*head_dim;
            float score = 0;
            for (int i = 0; i < head_dim; i++) score += qh[i] * kt[i];
            att[t] = score / sqrtf((float)head_dim);
        }
        softmax(att, pos + 1);

        // Use gdn_out buffer for attention output (2048 elements)
        float *out = s->gdn_out + h*head_dim;
        memset(out, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float *vt = s->vcache + loff + t*kvdim + kvh*head_dim;
            for (int i = 0; i < head_dim; i++) out[i] += att[t] * vt[i];
        }
    }

    // Apply sigmoid gate to attention output
    // Gate has same dimension as attention output: nheads * head_dim = 2048
    int attn_size = c->nheads * head_dim;  // 2048
    for (int i = 0; i < attn_size; i++) {
        s->gdn_out[i] = s->gdn_out[i] * (1.0f / (1.0f + expf(-gate[i])));
    }

    // Output projection
    // o_proj shape: [1024, 2048]
    matmul(s->x2, w->fa_o[layer], s->gdn_out, attn_size, dim);

    // Residual
    for (int i = 0; i < dim; i++) s->x[i] += s->x2[i];
}

// ============================================================================
// GDN (linear attention) forward
// ============================================================================

void forward_linear_attn(Transformer *t, int layer, int pos) {
    (void)pos;  // GDN is recurrent: it does not use the absolute position.
    Config *c = &t->c;
    State *s = &t->s;
    Weights *w = &t->w;

    int dim = c->dim;
    int nv = c->gdn_vheads;  // 16
    int kd = c->gdn_kdim;    // 128
    int vd = c->gdn_vdim;    // 128

    // Norm
    rmsnorm(s->x1, s->x, w->ln1[layer], dim);

    // QKV projection
    float *qkv = s->h;  // reuse h buffer [6144]
    matmul(qkv, w->gdn_qkv[layer], s->x1, dim, 6144);

    // Calculate GDN layer index for state
    int gdn_idx = 0;
    for (int i = 0; i < layer; i++) if (c->layer_types[i] == 0) gdn_idx++;
    
    // Conv1D: causal convolution over QKV
    // Weight shape: [6144, 1, 4] - depthwise conv with kernel 4
    // For each channel, we convolve with a kernel of size 4
    float *conv_state = s->conv_state + gdn_idx * 6144 * 4;
    float *conv_weight = w->gdn_conv[layer];
    
    // Shift conv state: [x0, x1, x2, x3] -> [x1, x2, x3, new_x]
    for (int i = 0; i < 6144; i++) {
        float *cs = conv_state + i * 4;  // state for this channel
        float *cw = conv_weight + i * 4; // weights for this channel (stored as [6144, 4])
        
        // Shift state left
        cs[0] = cs[1];
        cs[1] = cs[2];
        cs[2] = cs[3];
        cs[3] = qkv[i];  // new input
        
        // Apply convolution: sum over kernel (weight index k pairs with x[t-(K-1)+k])
        float sum = 0;
        for (int k = 0; k < 4; k++) {
            sum += cs[k] * cw[k];
        }
        // causal_conv1d_fn(..., activation=self.activation) with self.activation == "silu"
        qkv[i] = silu(sum);
    }
    
    // Z projection
    float *z = s->h1;  // [2048]
    matmul(z, w->gdn_z[layer], s->x1, dim, nv * vd);

    // A and B projections, then sigmoid
    float *ab = s->x2;  // [32]
    matmul(ab, w->gdn_a[layer], s->x1, dim, nv);
    matmul(ab + nv, w->gdn_b[layer], s->x1, dim, nv);
    
    // Only apply sigmoid to b (beta), a is used raw
    for (int i = nv; i < 2*nv; i++) {
        ab[i] = 1.0f / (1.0f + expf(-ab[i]));
    }

    // A is stored as log(A), need to exp
    float *A = w->gdn_A[layer];
    float *dt = w->gdn_dt[layer];

    float *state = s->gdn_state + gdn_idx * nv * kd * vd;

    // Process each head
    // Qwen3.5 GDN layout: qkv = [Q_all, K_all, V_all] sequential
    // Q: rows 0-2047 (16 heads * 128 dims each)
    // K: rows 2048-4095
    // V: rows 4096-6143
    // Total: 16 * 128 * 3 = 6144
    for (int h = 0; h < nv; h++) {
        float *q = qkv + h * kd;                    // Q in first section
        float *k = qkv + nv * kd + h * kd;          // K in second section (2048 offset)
        float *v = qkv + 2 * nv * kd + h * kd;      // V in third section (4096 offset)
        float *zh = z + h * vd;                      // Z from separate projection
        float *sh = state + h * kd * vd;

        // L2 normalize Q and K (critical for numerical stability)
        float q_norm = 0, k_norm = 0;
        for (int i = 0; i < kd; i++) { q_norm += q[i]*q[i]; k_norm += k[i]*k[i]; }
        q_norm = 1.0f / sqrtf(q_norm + 1e-6f);
        k_norm = 1.0f / sqrtf(k_norm + 1e-6f);
        for (int i = 0; i < kd; i++) { q[i] *= q_norm; k[i] *= k_norm; }
        
        // Scale Q by 1/sqrt(kd)
        float scale = 1.0f / sqrtf((float)kd);
        for (int i = 0; i < kd; i++) q[i] *= scale;

        // Decay and beta
        float a = ab[h];
        float b = ab[nv + h];
        // softplus(x) = log(1 + exp(x)) with PyTorch's threshold=20
        // g = -exp(A_log) * softplus(a + dt_bias)
        float sp = softplus_f(a + dt[h]);
        float g = -expf(A[h]) * sp;
        float gamma = expf(g);
        float beta = b;  // already sigmoid

        // Decay state
        for (int i = 0; i < kd * vd; i++) sh[i] *= gamma;

        // kv_mem = sum(state * k, dim=k) -> [v_head_dim]
        // In Python: kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        float kv_mem[128];
        for (int j = 0; j < vd; j++) {
            float sum = 0;
            for (int i = 0; i < kd; i++) sum += sh[i*vd + j] * k[i];
            kv_mem[j] = sum;
        }

        // delta = (v - kv_mem) * beta
        float delta[128];
        for (int j = 0; j < vd; j++) delta[j] = (v[j] - kv_mem[j]) * beta;

        // state += k.unsqueeze(-1) * delta.unsqueeze(-2)
        // In Python: state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        // k: [kd], delta: [vd]
        // outer product: [kd, vd]
        for (int i = 0; i < kd; i++) {
            for (int j = 0; j < vd; j++) sh[i*vd + j] += k[i] * delta[j];
        }

        // out = state @ q = sum(state * q.unsqueeze(-1), dim=k)
        // In Python: out = (state * q.unsqueeze(-1)).sum(dim=-2)
        float *out = s->gdn_out + h * vd;
        for (int j = 0; j < vd; j++) {
            float sum = 0;
            for (int i = 0; i < kd; i++) sum += sh[i*vd + j] * q[i];
            out[j] = sum;
        }

        // Gated RMSNorm (norm weight is shared across heads, shape [128])
        float ss = 0;
        for (int j = 0; j < vd; j++) ss += out[j] * out[j];
        ss = 1.0f / sqrtf(ss / vd + 1e-6f);
        for (int j = 0; j < vd; j++) out[j] = w->gdn_norm[layer][j] * ss * out[j] * silu(zh[j]);
    }

    // Output projection
    matmul(s->x2, w->gdn_out[layer], s->gdn_out, nv * vd, dim);

    // Residual
    for (int i = 0; i < dim; i++) s->x[i] += s->x2[i];
}

// ============================================================================
// MLP forward
// ============================================================================

void forward_mlp(Transformer *t, int layer) {
    Config *c = &t->c;
    State *s = &t->s;
    Weights *w = &t->w;
    
    int dim = c->dim;
    int ffndim = c->ffndim;
    
    rmsnorm(s->x1, s->x, w->ln2[layer], dim);
    
    matmul(s->h, w->gate[layer], s->x1, dim, ffndim);
    matmul(s->h1, w->up[layer], s->x1, dim, ffndim);
    
    // SwiGLU: gate * up
    for (int i = 0; i < ffndim; i++) s->h[i] = silu(s->h[i]) * s->h1[i];
    
    matmul(s->x1, w->down[layer], s->h, ffndim, dim);
    
    // Residual
    for (int i = 0; i < dim; i++) s->x[i] += s->x1[i];
}

// ============================================================================
// Forward pass
// ============================================================================

#if defined(QWEN35_TRACE)
static char *g_trace_path = NULL;
static FILE *g_trace_fp = NULL;
static int g_trace_enabled = 0;

static void trace_env_setup(void) {
    const char *p = getenv("QWEN35_TRACE");
    if (p && *p) {
        g_trace_path = strdup(p);
        g_trace_fp = fopen(g_trace_path, "wb");
        g_trace_enabled = g_trace_fp != NULL;
    }
}
static float rms_of(const float *x, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i]*x[i];
    return sqrtf(ss / (float)n);
}
#endif

float *forward(Transformer *t, int token, int pos) {
    Config *c = &t->c;
    State *s = &t->s;

#if defined(QWEN35_TRACE)
    if (!g_trace_enabled) trace_env_setup();
    int tracing = g_trace_enabled && pos == 0;
#endif

    // Embedding lookup
    memcpy(s->x, t->w.embed_tokens + token * c->dim, c->dim * sizeof(float));
#if defined(QWEN35_TRACE)
    if (tracing) {
        fprintf(g_trace_fp, "# layer=-1 (embedding) token=%d pos=%d stream_rms=%.8g\n",
                token, pos, rms_of(s->x, c->dim));
    }
#endif

    // Layers
    for (int l = 0; l < c->nlayers; l++) {
        if (c->layer_types[l] == 0) {
            forward_linear_attn(t, l, pos);
        } else {
            forward_full_attn(t, l, pos);
        }
        forward_mlp(t, l);
#if defined(QWEN35_TRACE)
        if (tracing) {
            // s->x2 = attention (GDN/full) residual delta, s->x1 = MLP residual
            // delta; s->x = running residual stream. (neither is overwritten
            // after this point, so all three are valid here)
            fprintf(g_trace_fp,
                    "layer=%2d type=%c attn_delta_rms=%.8g mlp_delta_rms=%.8g stream_rms=%.8g\n",
                    l, c->layer_types[l] ? 'F' : 'L',
                    rms_of(s->x2, c->dim), rms_of(s->x1, c->dim),
                    rms_of(s->x, c->dim));
        }
#endif
    }
    
    // Final norm
#if defined(QWEN35_TRACE)
    float pre_final_rms = tracing ? rms_of(s->x, c->dim) : 0.0f;
#endif
    rmsnorm(s->x, s->x, t->w.embed_norm, c->dim);
#if defined(QWEN35_TRACE)
    if (tracing) fprintf(g_trace_fp, "layer=23 (pre_final_norm) stream_rms=%.8g\n", pre_final_rms);
#endif
    
    // Logits
    // embed_tokens shape: [nvocab, dim] = [248320, 1024]
    // We need: logits = embed_tokens @ x, i.e., [nvocab, dim] @ [dim] = [nvocab]
    // matmul(o, W, x, n, d) computes o[i] = W[i,j] * x[j] for j=0..n, i=0..d
    matmul(s->logits, t->w.embed_tokens, s->x, c->dim, c->nvocab);

#if defined(QWEN35_TRACE)
    if (tracing) {
        // Top-5 logits by value (with index), and the argmax id.
        float lg[c->nvocab];
        for (int i = 0; i < c->nvocab; i++) lg[i] = s->logits[i];
        int top[5] = {0};
        for (int r = 0; r < 5; r++) {
            int best = -1; float bv = -1e30f;
            for (int i = 0; i < c->nvocab; i++) {
                if (s->logits[i] > bv) { bv = s->logits[i]; best = i; }
            }
            top[r] = best;
            s->logits[best] = -1e30f; // remove for next round
        }
        for (int i = 0; i < c->nvocab; i++) s->logits[i] = lg[i]; // restore
        fprintf(g_trace_fp,
                "layer=%d (final) post_final_norm_rms=%.8g\n",
                -2, rms_of(s->x, c->dim));
        for (int r = 0; r < 5; r++) {
            fprintf(g_trace_fp, "  top%d id=%d logit=%.8g\n",
                    r + 1, top[r], lg[top[r]]);
        }
    }
#endif

    return s->logits;
}

// ============================================================================
// Sampler
// ============================================================================

int sample_argmax(float *logits, int n) {
    int max_i = 0;
    float max_v = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_v) {
            max_v = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

// Temperature sampling
int sample_temp(float *logits, int n, float temp) {
    if (temp == 0.0f) return sample_argmax(logits, n);
    
    // Apply temperature
    for (int i = 0; i < n; i++) logits[i] /= temp;
    
    // Softmax
    softmax(logits, n);
    
    // Sample from distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < n; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    // Default paths. argv[1] is reserved for the prompt, so the model /
    // tokenizer paths can be overridden through the environment instead:
    //   QWEN35_MODEL=/path/model.safetensors QWEN35_VOCAB=/path/vocab.json
    const char *model_path = getenv("QWEN35_MODEL");
    if (!model_path || !*model_path)
        model_path = "/mnt/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";

    const char *vocab_path = getenv("QWEN35_VOCAB");
    if (!vocab_path || !*vocab_path)
        vocab_path = "/mnt/Qwen3.5-0.8B/vocab.json";

    // merges.txt sits next to vocab.json
    char merges_path[4096];
    {
        const char *slash = strrchr(vocab_path, '/');
        if (slash) {
            int dir_len = (int)(slash - vocab_path) + 1;
            if (dir_len > (int)sizeof(merges_path) - 32) dir_len = (int)sizeof(merges_path) - 32;
            memcpy(merges_path, vocab_path, dir_len);
            merges_path[dir_len] = '\0';
            strncat(merges_path, "merges.txt", sizeof(merges_path) - dir_len - 1);
        } else {
            snprintf(merges_path, sizeof(merges_path), "merges.txt");
        }
    }

    // The prompt is the command line argument:
    //   ./qwen35_engine "中国的首都是哪里？"
    // Fallbacks: QWEN35_PROMPT, then the built-in demo prompt below.
    const char *prompt_text = NULL;
    if (argc >= 2 && argv[1][0]) {
        prompt_text = argv[1];
    } else {
        const char *env_prompt = getenv("QWEN35_PROMPT");
        if (env_prompt && *env_prompt) prompt_text = env_prompt;
    }
    
    printf("Qwen3.5-0.8B C Inference Engine\n");
    printf("================================\n\n");
    
    // Initialize transformer
    Transformer tr;
    memset(&tr, 0, sizeof(Transformer));
    init_config(&tr.c);
    
    printf("Config: dim=%d, ffndim=%d, nlayers=%d, nheads=%d, nvocab=%d\n",
           tr.c.dim, tr.c.ffndim, tr.c.nlayers, tr.c.nheads, tr.c.nvocab);
    printf("Layer types: ");
    for (int i = 0; i < tr.c.nlayers; i++) {
        printf("%c", tr.c.layer_types[i] ? 'F' : 'L');
    }
    printf(" (L=linear, F=full)\n\n");
    
    // Load weights
    load_weights(&tr, model_path);
    
    malloc_state(&tr.s, &tr.c);
    
    // Initialize tokenizer
    BPETokenizer tokenizer;
    if (tokenizer_init(&tokenizer) != 0) {
        fprintf(stderr, "Warning: Failed to initialize tokenizer\n");
    }
    
    // Try to load vocab
    int tok_ready = 1;
    if (tokenizer_load_vocab(&tokenizer, vocab_path) != 0) {
        fprintf(stderr, "Warning: Failed to load vocab from %s\n", vocab_path);
        tok_ready = 0;
    }
    if (tok_ready && tokenizer_load_merges(&tokenizer, merges_path) != 0) {
        fprintf(stderr, "Warning: Failed to load merges from %s\n", merges_path);
        tok_ready = 0;
    }
    if (tok_ready) {
        tokenizer_load_special_tokens(&tokenizer);
    }
    if (!tok_ready && prompt_text) {
        fprintf(stderr, "Error: tokenizer unavailable, cannot encode the given prompt\n");
        return 1;
    }
    
    // ---- Build the prompt token sequence ----
    // Priority:
    //   1. explicit token ids from QWEN35_PROMPT_IDS (regression tests)
    //   2. command line / QWEN35_PROMPT text, wrapped in the chat template
    //   3. built-in demo prompt (same chat message as (2))
    const int MAX_PROMPT_TOKENS = 4096;
    int *prompt_ids = (int *)xmalloc(sizeof(int) * MAX_PROMPT_TOKENS);
    int n_prompt_ids = 0;
    int prompt_overridden = 0;

    const char *env_ids = getenv("QWEN35_PROMPT_IDS");
    if (env_ids && *env_ids) {
        const char *p = env_ids;
        while (*p && n_prompt_ids < MAX_PROMPT_TOKENS) {
            prompt_ids[n_prompt_ids++] = (int)strtol(p, (char **)&p, 10);
            while (*p == ',' || *p == ' ') p++;
        }
        if (n_prompt_ids > 0) prompt_overridden = 1;
    }

    if (!prompt_overridden) {
        // The chat template renders the message and appends the generation
        // prompt:
        //   <|im_start|>user\n{prompt}<|im_end|>\n
        //   <|im_start|>assistant\n<think>\n\n</think>\n\n
        // (enable_thinking defaults to false for this model)
        const char *user_text = prompt_text ? prompt_text : "中国的首都是哪里？";
        char *chat_prompt = (char *)xmalloc(strlen(user_text) + 512);
        sprintf(chat_prompt,
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                user_text);

        n_prompt_ids = tokenizer_encode(&tokenizer, chat_prompt, prompt_ids,
                                        MAX_PROMPT_TOKENS);
        if (n_prompt_ids <= 0) {
            die("tokenizer_encode failed");
        }

        printf("Prompt: \"%s\"\n", user_text);
        printf("Token ids (%d):", n_prompt_ids);
        for (int i = 0; i < n_prompt_ids && i < 64; i++) {
            printf(" %d", prompt_ids[i]);
        }
        if (n_prompt_ids > 64) printf(" ...");
        printf("\n");
        free(chat_prompt);
    } else {
        printf("Prompt: (from QWEN35_PROMPT_IDS, %d tokens)\n", n_prompt_ids);
    }
    printf("Output: ");
    fflush(stdout);
    
    int token = prompt_ids[0];
    int n_generate = 150;
    int max_pos = n_prompt_ids + n_generate;
    if (max_pos > tr.c.nctx) max_pos = tr.c.nctx;
    
    // ---- Prefill: process all prompt tokens, timed ----
    double t_prefill_start = now_sec();
    float *logits = forward(&tr, token, 0);
    for (int pos = 1; pos < n_prompt_ids; pos++) {
        token = prompt_ids[pos];
        logits = forward(&tr, token, pos);
    }
    double t_prefill_end = now_sec();
    double prefill_sec = t_prefill_end - t_prefill_start;
    fflush(stdout);

    // Optional logits dump for numeric verification against the HF reference:
    //   QWEN35_DUMP_LOGITS=/path/out.bin ./qwen35_engine
    // Writes nvocab float32 logits of the last prompt position.
    {
        const char *dump_path = getenv("QWEN35_DUMP_LOGITS");
        if (dump_path) {
            FILE *df = fopen(dump_path, "wb");
            if (df) {
                fwrite(logits, sizeof(float), tr.c.nvocab, df);
                fclose(df);
                fprintf(stderr, "[dump] wrote %d logits to %s\n", tr.c.nvocab, dump_path);
            } else {
                fprintf(stderr, "[dump] cannot open %s\n", dump_path);
            }
        }
    }
    
    // ---- Decode: greedy generation ----
    // Timing accumulates only around forward() so stdout I/O doesn't pollute
    // the throughput numbers.
    int pos = n_prompt_ids;
    int n_generated = 0;
    double decode_sec = 0.0;
    int hit_eos = 0;
    while (pos < max_pos) {
        int next = sample_argmax(logits, tr.c.nvocab);
        
        // Stop at EOS without printing the special token itself
        if (next == SPECIAL_TOKEN_EOS) {
            hit_eos = 1;
            break;
        }
        n_generated++;
        
        // Forward next token
        token = next;
        {
            double t0 = now_sec();
            logits = forward(&tr, token, pos);
            decode_sec += now_sec() - t0;
        }
        pos++;
        
        // Decode and print
        char decoded[256];
        int tid = next;
        int len = tokenizer_decode(&tokenizer, &tid, 1, decoded, sizeof(decoded));
        if (len > 0) {
            printf("%s", decoded);
        } else {
            printf("[%d]", next);
        }
        fflush(stdout);
    }
    
    // ---- Throughput stats ----
    printf("\n\n-------------------\n");
    printf("Prefill: %d tokens in %.3f s  (%.2f tok/s)\n",
           n_prompt_ids, prefill_sec,
           prefill_sec > 0 ? n_prompt_ids / prefill_sec : 0.0);
    printf("Generate: %d tokens in %.3f s  (%.2f tok/s)\n",
           n_generated, decode_sec,
           decode_sec > 0 ? n_generated / decode_sec : 0.0);
    printf("%s\n", hit_eos ? "Stop reason: EOS" : "Stop reason: max tokens reached");
    
    // Cleanup
    tokenizer_free(&tokenizer);
    safetensors_close(tr.sf);
    
    return 0;
}