#!/home/wisedu/laya/bin/python
"""
exp_decay.py -- real-weight experiments for Qwen3.5-0.8B layer 0 GatedDeltaNet.

Answers SPEC questions 4 (g shape / softplus / gamma sign & range), 5 (repeat_interleave),
6 (gated RMSNorm order), 7 (A_log), 8 (actual gamma range), 2 (conv), 3 (l2norm),
and cross-checks a hand-written GDN layer against the REAL transformers module.

Run: /home/wisedu/laya/bin/python exp_decay.py
"""
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/mnt/Qwen3.5-0.8B"
ST = f"{MODEL_DIR}/model.safetensors-00001-of-00001.safetensors"
PFX = "model.language_model."
L0 = f"{PFX}layers.0."
CFG = json.load(open(f"{MODEL_DIR}/config.json"))
TC = CFG["text_config"]

f = safe_open(ST, framework="pt")
def W(name):
    return f.get_tensor(PFX + name).float()

print("=" * 78)
print("[Q7/Q8] real A_log / dt_bias of layer 0 (fp32 in checkpoint)")
A_log = W("layers.0.linear_attn.A_log")
dt_bias = W("layers.0.linear_attn.dt_bias")
print("  A_log  =", [round(x, 6) for x in A_log.tolist()])
print("  dt_bias=", [round(x, 6) for x in dt_bias.tolist()])
print("  A_log.exp() =", [round(x, 6) for x in A_log.exp().tolist()],
      "  (range %.4f .. %.4f)" % (A_log.exp().min(), A_log.exp().max()))
print("  => A_log holds log(A) with A in [0.01,16] (modeling_qwen3_5.py:535-537, 938-942);")
print("     the code uses A = A_log.exp() (line 619: -self.A_log.float().exp() * F.softplus(...)).")
print("     NOTE the decoder DeltaNet uses exp(A_log) = A, i.e. A itself, NOT exp of a log already")

# ---------------------------------------------------------------- real inputs
print("\n[setup] building layer-0 inputs from real embeddings (text-only, no vision)")
emb = f.get_tensor(PFX + "embed_tokens.weight").float()          # [248320, 1024]
ln_w = W("layers.0.input_layernorm.weight")                      # [1024], zero-centred
print("  input_layernorm.weight: mean=%.5f std=%.5f min=%.5f max=%.5f"
      % (ln_w.mean(), ln_w.std(), ln_w.min(), ln_w.max()))
print("  -> zero-centred, so the effective scale is (1 + w)  (Qwen3_5RMSNorm, line 854)")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    print("  tokenizer ok, ids =", ids.tolist())
except Exception as e:  # pragma: no cover
    print("  tokenizer unavailable (%r); falling back to fixed ids" % e)
    ids = torch.tensor([785, 6722, 315, 9625, 374])
h = emb[ids]                                                     # [T,1024] already masked-token free
T = h.shape[0]

def rmsnorm_1plus(x, w, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) * (1.0 + w.float())

hn = rmsnorm_1plus(h, ln_w)

# ------------------------------------------------- Q1: split order
qkv_w = W("layers.0.linear_attn.in_proj_qkv.weight")
print("\n[Q1] in_proj_qkv.weight shape =", tuple(qkv_w.shape))
mixed = hn @ qkv_w.t()                                           # [T, 6144]
key_dim, value_dim = 128 * 16, 128 * 16
qs, ks, vs = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
print("  torch.split(mixed_qkv, [key_dim, key_dim, value_dim]) with key_dim=%d value_dim=%d"
      % (key_dim, value_dim))
print("  -> CONTIGUOUS BLOCKS: [0:2048]=Q, [2048:4096]=K, [4096:6144]=V. NOT interleaved.")
print("  key_dim = head_k_dim*num_k_heads = %d*%d = %d (line 512); conv_dim = 2*key_dim+value_dim = %d (line 521)"
      % (128, 16, key_dim, 2 * key_dim + value_dim))

# ------------------------------------------------- Q2: conv1d
cw = f.get_tensor(L0 + "linear_attn.conv1d.weight").float()      # [6144,1,4]
print("\n[Q2] conv1d.weight shape =", tuple(cw.shape), " bias =", "absent (bias=False, line 525)")
print("  kernel_size = linear_conv_kernel_dim =", TC["linear_conv_kernel_dim"],
      " groups = channels = 6144 (depthwise), padding = K-1 =", TC["linear_conv_kernel_dim"] - 1)
print("  activation = config.hidden_act =", TC["hidden_act"], "(passed as activation=self.activation, line 594)")
x = mixed.t().unsqueeze(0)                                       # [1, 6144, T]
conv_out = F.conv1d(x, cw, None, padding=3, groups=6144)[:, :, :T]
conv_out = F.silu(conv_out)
left_pad = F.conv1d(F.pad(x, (3, 0)), cw, None, padding=0, groups=6144)[:, :, :T]
left_pad = F.silu(left_pad)
print("  padding=K-1 in F.conv1d is LEFT padding;  max|left_pad - ref_pad| =",
      (left_pad - conv_out).abs().max().item())
print("  => silu is applied IMMEDIATELY after conv (activation passed into causal_conv1d_fn,")
print("     which does ACT2FN[activation](out) at line 287-288), BEFORE the Q/K/V split.")

# ------------------------------------------------- Q5: repeat_interleave
nkh, nvh = TC["linear_num_key_heads"], TC["linear_num_value_heads"]
print("\n[Q5] linear_num_k_heads =", nkh, " linear_num_v_heads =", nvh,
      " -> num_v_heads // num_k_heads =", nvh // nkh)
print("  condition `if self.num_v_heads // self.num_k_heads > 1` (line 620) is",
      "FALSE -> repeat_interleave NOT executed for this model" if nvh // nkh <= 1 else "TRUE")

# ------------------------------------------------- Q4: g / softplus / beta
b_w, a_w = W("layers.0.linear_attn.in_proj_b.weight"), W("layers.0.linear_attn.in_proj_a.weight")
beta = (hn @ b_w.t()).sigmoid()
a = hn @ a_w.t()
z_pre = a + dt_bias
z = F.softplus(z_pre)
print("\n[Q4c] F.softplus(a.float() + dt_bias) -- torch default beta=1.0, threshold=20.0")
print("  (a+dt_bias) range: [%.4f, %.4f]  -> max %.4f %s 20, so the threshold branch is %s"
      % (z_pre.min(), z_pre.max(), z_pre.max(),
         "<" if z_pre.max() < 20 else ">=", "NEVER taken" if z_pre.max() < 20 else "taken"))
print("  softplus == log1p(exp(x)) here: max|softplus - log1p(exp(x))| =",
      (z - torch.log1p(torch.exp(z_pre))).abs().max().item())
print("  manual log1p(exp()) vs torch, max|diff| =", (torch.log1p(torch.exp(z_pre)) - z).abs().max().item())
g = -A_log.exp() * z
gamma = g.exp()
print("\n[Q4b/Q8] g = -exp(A_log) * softplus(a + dt_bias)  (line 619)")
print("  g        range: [%.6f, %.6f]   (must be <= 0)" % (g.min(), g.max()))
print("  gamma=exp(g) range: [%.8f, %.8f]  -> strictly inside (0,1): %s"
      % (gamma.min(), gamma.max(), bool((gamma > 0).all() and (gamma < 1).all())))
print("  per-token gamma mean = %.6f" % gamma.mean())
print("  gamma shape =", tuple(gamma.shape),
      " = [batch, seq, num_v_heads] -> ONE SCALAR PER (token, v-head), shared over k-dim")
print("  => the decay is NOT per-k-channel; a single scalar multiplies the whole [k_dim,v_dim] state")

# ------------------------------------------------- Q3: l2norm
print("\n[Q3] l2norm (modeling_qwen3_5.py:294-297):")
print("   inv_norm = torch.rsqrt((x*x).sum(dim=-1,keepdim=True) + eps); return x * inv_norm")
print("   eps = 1e-6, hard-coded at the call sites (lines 344-345 / 464-465), i.e. rsqrt(sum+1e-6) form.")
print("   q is THEN scaled: query = query / (query.shape[-1] ** 0.5)  (line 468), head_k_dim=128 -> /sqrt(128)")

# ------------------------------------------------- Q6: gated RMSNorm
nw = W("layers.0.linear_attn.norm.weight")
print("\n[Q6] Qwen3_5RMSNormGated (line 218-234), weight shape =", tuple(nw.shape),
      "== head_v_dim (per-head, ONE shared vector of length head_v_dim)")
print("   weight stats mean=%.5f std=%.5f min=%.5f max=%.5f  (init ones, so NOT zero-centred:"
      " line 221 vs Qwen3_5RMSNorm line 845 which is zeros/(1+w))"
      % (nw.mean(), nw.std(), nw.min(), nw.max()))
print("   ORDER (lines 228-232): variance = mean(x^2) over the LAST dim (head_v_dim);")
print("                          x = x * rsqrt(variance + eps);  x = weight * x;")
print("                          x = x * silu(gate)   <-- NORM FIRST, THEN gate")
print("   eps = config.rms_norm_eps =", TC["rms_norm_eps"])

# --------------------------------------- Q12/Q13/Q14: full-attention layer facts
print("\n[Q12] q_norm/k_norm are applied BEFORE RoPE: lines 792-797 --")
print("      query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1,2)")
print("      key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1,2)")
print("      ... then apply_rotary_pos_emb(query_states, key_states, cos, sin)  (line 797)")
print("      both norms have weight shape [256] == head_dim (config head_dim=256, rms_norm_eps=1e-6)")

print("\n[Q13] scaling = head_dim**-0.5 = %.8f (line 758, head_dim=%d)"
      % (TC["head_dim"] ** -0.5, TC["head_dim"]))
print("      num_key_value_groups = 8//2 =", TC["num_attention_heads"] // TC["num_key_value_heads"])
print("      repeat_kv (line 712-721): [b, nkv, s, hd] -> [:, :, None].expand(b, nkv, n_rep, s, hd)")
print("      -> reshape(b, nkv*n_rep, s, hd).  Because the expanded axis ORDER is (nkv, n_rep),")
print("      flattened head index h maps to kv head  kv = h // n_rep  (equivalently h // groups).")

# ------------------------------------------------- Q14
qp = W("layers.3.self_attn.q_proj.weight") if False else f.get_tensor(
    PFX + "layers.3.self_attn.q_proj.weight").float()
print("\n[Q14] q_proj.weight shape =", tuple(qp.shape),
      " -> out_features = 2*num_heads*head_dim = 2*8*256 = 4096")
print("      query_states, gate = torch.chunk(self.q_proj(h).view(..., head_dim*2), 2, dim=-1) (line 787-789)")
print("      -> SPLIT ALONG THE HEAD-DIM AXIS, i.e. per head the 512 values are [q(256) | gate(256)].")
print("      gate is applied with torch.sigmoid: attn_output * torch.sigmoid(gate) (line 818)")

# ================= hand-written GDN vs the REAL module =================
print("\n" + "=" * 78)
print("[FULL LAYER CHECK] hand-written layer-0 GDN  vs  real Qwen3_5GatedDeltaNet")
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5TextConfig

cfg = Qwen3_5TextConfig(**TC)
ref_mod = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).float().eval()
sd = {}
for n in ["in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
          "conv1d.weight", "dt_bias", "A_log", "norm.weight", "out_proj.weight"]:
    sd[n] = f.get_tensor(L0 + "linear_attn." + n).float()
ref_mod.load_state_dict(sd, strict=True)
print("  loaded real weights into the module (strict=True) -- OK")

with torch.no_grad():
    ref_out = ref_mod(hn.unsqueeze(0)).squeeze(0)   # [T,1024]

def manual_gdn(hid, sd, use_l2norm=True, state_order="pre_decay_read", norm_before_gate=True):
    T = hid.shape[0]
    mixed = (hid @ sd["in_proj_qkv.weight"].t()).t().unsqueeze(0)      # [1,6144,T]
    c = F.conv1d(mixed, sd["conv1d.weight"], None, padding=3, groups=6144)[:, :, :T]
    c = F.silu(c)
    m = c.transpose(1, 2)[0]                                            # [T,6144]
    q, k, v = torch.split(m, [2048, 2048, 2048], dim=-1)
    q = q.reshape(T, 16, 128)
    k = k.reshape(T, 16, 128)
    v = v.reshape(T, 16, 128)
    z = (hid @ sd["in_proj_z.weight"].t()).reshape(T, 16, 128)
    beta = (hid @ sd["in_proj_b.weight"].t()).sigmoid()
    a = hid @ sd["in_proj_a.weight"].t()
    g = -sd["A_log"].exp() * F.softplus(a + sd["dt_bias"])
    if use_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q / (128 ** 0.5)
    S = torch.zeros(16, 128, 128)
    out = torch.zeros(T, 16, 128)
    for i in range(T):
        gm = g[i].exp()[:, None, None]
        if state_order == "pre_decay_read":
            S = S * gm
            kv_mem = (S * k[i].unsqueeze(-1)).sum(dim=-2)
            delta = (v[i] - kv_mem) * beta[i].unsqueeze(-1)
            S = S + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
        else:
            kv_mem = (S * k[i].unsqueeze(-1)).sum(dim=-2)
            delta = (v[i] - kv_mem) * beta[i].unsqueeze(-1)
            S = S * gm + k[i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[i] = (S * q[i].unsqueeze(-1)).sum(dim=-2)
    if norm_before_gate:
        var = out.pow(2).mean(-1, keepdim=True)
        n = out * torch.rsqrt(var + 1e-6) * sd["norm.weight"]
    else:
        n = out * sd["norm.weight"] * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-6)
    n = n * F.silu(z)
    return n.reshape(T, 2048) @ sd["out_proj.weight"].t()

man = manual_gdn(hn, sd)
man_bad = manual_gdn(hn, sd, state_order="post_decay_read")
print("  max|manual(correct order) - real module|      = %.3e" % (man - ref_out).abs().max().item())
print("  max|manual(WRONG  order) - real module|      = %.3e" % (man_bad - ref_out).abs().max().item())
print("  max|manual(no l2norm)|...                    = %.3e"
      % (manual_gdn(hn, sd, use_l2norm=False) - ref_out).abs().max().item())
print("  ref_out[0,:4] =", [round(x, 6) for x in ref_out[0, :4].tolist()])
print("  man_out[0,:4] =", [round(x, 6) for x in man[0, :4].tolist()])
print("  relative error %.3e" % ((man - ref_out).abs().max() / ref_out.abs().max()))
print("=" * 78)
