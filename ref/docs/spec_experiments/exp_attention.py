#!/home/wisedu/laya/bin/python
"""
exp_attention.py -- real-weight checks for the Qwen3.5-0.8B full_attention layers (layer 3)
and for the embedding / tied lm_head / MLP.

Answers SPEC Q9-Q14 (numerically) and Q16.
Run: /home/wisedu/laya/bin/python exp_attention.py
"""
import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_DIR = "/mnt/Qwen3.5-0.8B"
ST = f"{MODEL_DIR}/model.safetensors-00001-of-00001.safetensors"
PFX = "model.language_model."
CFG = json.load(open(f"{MODEL_DIR}/config.json"))
TC = CFG["text_config"]
f = safe_open(ST, framework="pt")


def T(name):
    return f.get_tensor(PFX + name).float()


print("=" * 78)
# ---------------------------------------------------------------- Q16 + MLP
print("[Q16] checkpoint keys: lm_head present?",
      any("lm_head" in k for k in f.keys()))
print("      embed_tokens:", tuple(f.get_tensor(PFX + "embed_tokens.weight").shape))
print("      -> no lm_head tensor; _tied_weights_keys = {'lm_head.weight': 'model.embed_tokens.weight'}")
print("         (modeling_qwen3_5.py:1665) and config tie_word_embeddings =", CFG["tie_word_embeddings"])
print("      model.language_model.norm.weight:", tuple(T("norm.weight").shape))
print("      lm_head = nn.Linear(hidden, vocab, bias=False) (line 1676); logits = x @ W^T")
print("      (nn.Linear stores weight as [out_features, in_features] = [248320, 1024])")

mlp0 = [(k, tuple(f.get_tensor(PFX + "layers.0.mlp." + k).shape))
        for k in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]]
print("\n[MLP] layer0 tensors:", mlp0)
print("      SwiGLU: down_proj( silu(gate_proj(x)) * up_proj(x) )   (Qwen3_5MLP, see modeling file)")
print("      ACT2FN[config.hidden_act] with hidden_act =", TC["hidden_act"])

# ---------------------------------------------------------------- Q9/Q10/Q12/Q13/Q14
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5RMSNorm,
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

cfg = Qwen3_5TextConfig(**TC)
rope = Qwen3_5TextRotaryEmbedding(cfg).eval()

emb = T("embed_tokens.weight")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
except Exception:
    ids = torch.tensor([785, 6722, 315, 9625, 374, 369])
N = ids.shape[0]
h = emb[ids]

ln_w = T("layers.3.input_layernorm.weight")
def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
hn = rms(h, ln_w)

# pure-text position ids -> rotary
pos = torch.arange(N).view(1, 1, -1).expand(4, 1, -1)[1:]
with torch.no_grad():
    cos, sin = rope(torch.zeros(1, N, TC["hidden_size"]), pos)
print("\n[Q9/Q10] rotary_dim (2*len(inv_freq)) =", int(rope.inv_freq.shape[0] * 2),
      " cos.shape =", tuple(cos.shape), " (last dim = 2*rotary_dim)")
print("      cos[0,1,:6] =", [round(x, 8) for x in cos[0, 1, :6].tolist()])
print("      sin[0,1,:6] =", [round(x, 8) for x in sin[0, 1, :6].tolist()])

qn_w, kn_w = T("layers.3.self_attn.q_norm.weight"), T("layers.3.self_attn.k_norm.weight")
print("\n[Q12] q_norm.weight shape", tuple(qn_w.shape),
      " mean=%.5f min=%.5f max=%.5f" % (qn_w.mean(), qn_w.min(), qn_w.max()))
print("      kn_w stats mean=%.5f min=%.5f max=%.5f" % (kn_w.mean(), kn_w.min(), kn_w.max()))
print("      NOTE Qwen3_5RMSNorm init is torch.zeros + forward multiplies by (1.0 + weight)")
print("           (lines 845 & 854). These are real trained values != 0, but the +1 convention")
print("           is what matters: effective scale = 1 + w.  (Same for input/post_attention layernorm.)")

# ---------------- hand-written attention vs the real module
sd = {n: f.get_tensor(PFX + "layers.3.self_attn." + n).float()
      for n in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                "q_norm.weight", "k_norm.weight"]}
ref = Qwen3_5Attention(cfg, layer_idx=3).float().eval()
ref.load_state_dict(sd, strict=True)

H, HKV, D = TC["num_attention_heads"], TC["num_key_value_heads"], TC["head_dim"]
scale = D ** -0.5

# A standalone Qwen3_5Attention call with attention_mask=None does NOT add any causal mask
# (the mask is built by Qwen3_5TextModel.forward -> create_causal_mask and passed in), so build
# the standard additive causal mask ourselves and hand it to the module -- this is what the real
# model does for the full_attention layers.
causal_additive = torch.full((1, 1, N, N), float("-inf"))
causal_additive.masked_fill_(torch.tril(torch.ones(N, N, dtype=torch.bool)), 0.0)

with torch.no_grad():
    ref_out, _ = ref(hn.unsqueeze(0), (cos, sin), causal_additive)


def manual_attn(hn, sd, rope_before_norm=False, mapping="floor", scale_=scale, causal=True,
                head_contig=True, diag=False):
    N = hn.shape[0]
    if head_contig:
        qg = (hn @ sd["q_proj.weight"].t()).view(N, H, 2 * D)
        q_raw, gate = qg[..., :D], qg[..., D:]
    else:
        qp = (hn @ sd["q_proj.weight"].t()).view(N, 2, H, D)   # wrong: [q-block | gate-block]
        q_raw, gate = qp[:, 0], qp[:, 1]
    k_raw = (hn @ sd["k_proj.weight"].t()).view(N, HKV, D)
    if rope_before_norm:
        # deliberately wrong variant: RoPE on the RAW projection, norm applied afterwards
        q, k = apply_rotary_pos_emb(q_raw.transpose(0, 1).unsqueeze(0),
                                    k_raw.transpose(0, 1).unsqueeze(0), cos, sin)
        q = rms(q.transpose(1, 2), sd["q_norm.weight"]).transpose(1, 2)
        k = rms(k.transpose(1, 2), sd["k_norm.weight"]).transpose(1, 2)
    else:
        q = rms(q_raw, sd["q_norm.weight"]).transpose(0, 1).unsqueeze(0)   # [1,H,N,D]
        k = rms(k_raw, sd["k_norm.weight"]).transpose(0, 1).unsqueeze(0)   # [1,nkv,N,D]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    v = (hn @ sd["v_proj.weight"].t()).view(N, HKV, D).transpose(0, 1).unsqueeze(0)
    # repeat_kv
    groups = H // HKV
    if mapping == "floor":
        kidx = [hh // groups for hh in range(H)]
    else:
        kidx = [hh % HKV for hh in range(H)]
    k2 = torch.stack([k[0, i] for i in kidx]).unsqueeze(0)
    v2 = torch.stack([v[0, i] for i in kidx]).unsqueeze(0)
    att = (q @ k2.transpose(-1, -2)) * scale_
    if causal:
        m = torch.full((N, N), float("-inf"))
        m.masked_fill_(torch.tril(torch.ones(N, N, dtype=torch.bool)), 0.0)
        att = att + m
    att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
    o = att @ v2
    o = o.transpose(1, 2).reshape(N, H * D)
    o = o * torch.sigmoid(gate.reshape(N, H * D))
    return o @ sd["o_proj.weight"].t()


man = manual_attn(hn, sd)
print("\n[Q12/Q13/Q14] hand-written full attention vs real Qwen3_5Attention:")
print("   max|correct - real|                     = %.3e" % (man - ref_out).abs().max().item())
print("   max|rope-before-norm (WRONG) - real|    = %.3e"
      % (manual_attn(hn, sd, rope_before_norm=True) - ref_out).abs().max().item())
print("   max|gate-block split (WRONG) - real|    = %.3e"
      % (manual_attn(hn, sd, head_contig=False) - ref_out).abs().max().item())
print("   max|h%%nkv mapping (WRONG) - real|        = %.3e"
      % (manual_attn(hn, sd, mapping="mod") - ref_out).abs().max().item())
print("   max|no causal mask (WRONG) - real|      = %.3e"
      % (manual_attn(hn, sd, causal=False) - ref_out).abs().max().item())
print("   max|scale=1.0 (WRONG) - real|           = %.3e"
      % (manual_attn(hn, sd, scale_=1.0) - ref_out).abs().max().item())

# ------------------------------------------------ whole-model logits sanity
print("\n[Q16] end-to-end: real model forward, last-token logits argmax")
try:
    from transformers import AutoModelForCausalLM
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    mdl.eval()
    with torch.no_grad():
        out = mdl(input_ids=torch.tensor([ids.tolist()]), use_cache=False,
                  output_hidden_states=True)
    lg = out.logits[0, -1]
    am = int(lg.argmax())
    print("   logits shape", tuple(out.logits.shape),
          " argmax =", am, "->", repr(tok.decode([am])))
    print("   lm_head weight tied to embed_tokens?",
          mdl.lm_head.weight.data_ptr() == mdl.model.embed_tokens.weight.data_ptr())
    print("   tie check (allclose):", bool(torch.allclose(mdl.lm_head.weight, mdl.model.embed_tokens.weight)))
    # manual logits from the final norm (hidden_states[-1] is already after model.norm)
    hn_last = out.hidden_states[-1][0, -1]
    man_am = int((hn_last @ mdl.lm_head.weight.t()).argmax())
    print("   (manual) final_norm(x) @ W^T argmax =", man_am, "== model argmax:", man_am == am)
    # prove the WRONG orientation fails: W @ x is not even shape-compatible here
    print("   W is [vocab, hidden] =", tuple(mdl.lm_head.weight.shape),
          "so logits = x @ W.T  (NOT W @ x, which would need W as [hidden, vocab])")
except Exception as e:
    print("   full-model forward skipped:", repr(e))
print("=" * 78)
