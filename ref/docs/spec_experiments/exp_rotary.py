#!/home/wisedu/laya/bin/python
"""
exp_rotary.py -- measures the REAL rotary_dim / cos / sin for Qwen3.5-0.8B text (no image tokens).

Answers SPEC questions 9, 10, 11 by RUNNING the real Qwen3_5TextRotaryEmbedding and by
reproducing transformers' position_ids construction for pure-text input
(modeling_qwen3_5.py:1260-1271).

Run: /home/wisedu/laya/bin/python exp_rotary.py
"""
import json

import torch

MODEL_DIR = "/mnt/Qwen3.5-0.8B"
CFG = json.load(open(f"{MODEL_DIR}/config.json"))
TC = CFG["text_config"]

print("=" * 78)
print("Q9  config.rope_parameters =", json.dumps(TC["rope_parameters"], sort_keys=True))
print("    head_dim =", TC["head_dim"], " num_attention_heads =", TC["num_attention_heads"])
print("    partial_rotary_factor =", TC["rope_parameters"]["partial_rotary_factor"])
print("    rope_theta =", TC["rope_parameters"]["rope_theta"],
      " rope_type =", TC["rope_parameters"]["rope_type"])
print("    mrope_section =", TC["rope_parameters"]["mrope_section"],
      " mrope_interleaved =", TC["rope_parameters"]["mrope_interleaved"])
print("    hidden_size =", TC["hidden_size"], "rms_norm_eps =", TC["rms_norm_eps"])
print("    num_key_value_heads =", TC["num_key_value_heads"],
      "vocab_size =", TC["vocab_size"], "tie_word_embeddings =", CFG["tie_word_embeddings"])
print("=" * 78)

# ---------------------------------------------------------------- real module
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextConfig,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)

cfg = Qwen3_5TextConfig(**TC)
rope = Qwen3_5TextRotaryEmbedding(cfg)
rope.eval()

inv = rope.inv_freq
print("\n[Q9] MEASURED via the real module (not inferred):")
print("    rope.inv_freq.shape       =", tuple(inv.shape))
print("    rope.attention_scaling    =", rope.attention_scaling)
print("    rope.rope_type            =", rope.rope_type)
print("    rope.mrope_section        =", rope.mrope_section)
rotary_dim = int(inv.shape[-1] * 2)
print("    => rotational dim (2 * len(inv_freq)) = ", rotary_dim)
print("       check against head_dim*partial_rotary_factor =",
      int(TC["head_dim"] * TC["rope_parameters"]["partial_rotary_factor"]))
print("    => head_dim =", TC["head_dim"], " so UN-rotated tail length =",
      TC["head_dim"] - rotary_dim)
print("    inv_freq[:6] =", [round(x, 8) for x in inv[:6].tolist()])
print("    inv_freq[-3:]= ", [round(x, 10) for x in inv[-3:].tolist()])

# ---------------------------------------------------- pure-text position_ids
# modeling_qwen3_5.py:1260-1271 for a fresh text-only forward, bs=1, N=3 tokens, no cache:
#   position_ids = arange(N).view(1,1,-1).expand(4, bs, -1)
#   -> text_position_ids = position_ids[0];  rotary gets position_ids[1:]  (3 grids, all identical)
N, BS = 3, 1
pos_ids = torch.arange(N).view(1, 1, -1).expand(4, BS, -1)
text_pos = pos_ids[0]
rope_pos = pos_ids[1:]
print("\n[Q10] pure-text position_ids:")
print("    full position_ids shape =", tuple(pos_ids.shape), " (4, bs, N)")
print("    text_position_ids shape =", tuple(text_pos.shape), " values =", text_pos.tolist())
print("    given to rotary (=[1:]) shape =", tuple(rope_pos.shape),
      " values (grid0/1/2) =", [rope_pos[i][0].tolist() for i in range(3)])

with torch.no_grad():
    cos, sin = rope(torch.zeros(1, N, TC["hidden_size"]), rope_pos)
print("    returned cos.shape =", tuple(cos.shape), " sin.shape =", tuple(sin.shape))
print("    cos.shape[-1] =", cos.shape[-1], "== rotary_dim = 2*len(inv_freq) = %d" % rotary_dim)
print("         (len(inv_freq)=%d because dim=int(head_dim*partial_rotary_factor)=64 and" % inv.shape[-1])
print("          inv_freq = 1/base**(arange(0,64,2)/64) -> 32 values).  recomposition_frequencies")
print("         (line 213) ends with cat((freqs_thw, freqs_thw), -1), which is what turns the 32")
print("         frequencies into the 64-wide cos/sin that apply_rotary_pos_emb consumes.")
print("         So exactly the FIRST 64 of the 256 head dims are rotated; 192 pass through untouched.")

print("\n[Q10] first 8 cos values for positions 0,1,2 (head 0):")
print("      pos | cos[0:8]")
for p in range(N):
    print("       %d  |" % p, [round(x, 8) for x in cos[0, p, :8].tolist()])
print("      first 8 sin values for positions 0,1,2 (head 0):")
for p in range(N):
    print("       %d  |" % p, [round(x, 8) for x in sin[0, p, :8].tolist()])

# verify the cat(freq, freq) duplication claim: cos[..., :32] == cos[..., 32:64]
half = rotary_dim // 2
dup = bool(torch.equal(cos[..., :half], cos[..., half:rotary_dim]))
print("\n[Q10] recomposition_frequencies ends with torch.cat((freqs_thw, freqs_thw), dim=-1)")
print("      so cos[..., :32] == cos[..., 32:64] ?  ->", dup)
print("      and for pure text all 3 mrope grids are identical, so the H/W overwrite")
print("      in recomposition_frequencies (lines 204-213) changes NOTHING numerically:")
flat_only = (rope_pos[0] == rope_pos[1]).all() and (rope_pos[1] == rope_pos[2]).all()
print("      grid0==grid1==grid2 ?", bool(flat_only))

# manual re-derivation, no mrope at all, must match
manual_freqs = (inv.float()[None, None, :, None] * rope_pos[0:1, :, None, :].float()).transpose(2, 3)
mcos_plain = torch.cat([manual_freqs.cos(), manual_freqs.cos()], dim=-1)
msin_plain = torch.cat([manual_freqs.sin(), manual_freqs.sin()], dim=-1)
print("      max|cos_real - plain_cat(cos,cos)| =", (mcos_plain - cos.float()).abs().max().item())
print("      max|sin_real - plain_cat(sin,sin)| =", (msin_plain - sin.float()).abs().max().item())
print("      => for pure text the effective cos/sin are simply cos(pos*inv_freq) duplicated.")

# ------------------------------------------------- Q11/rotate_half semantics
print("\n[Q11] rotate_half (line 666-670): x1=x[...,:d/2], x2=x[...,d/2:], return cat(-x2, x1)")
x = torch.arange(8, dtype=torch.float32)
print("      rotate_half", x.tolist(), "=", rotate_half(x).tolist())
print("      => PAIRS ACROSS THE HALVES: (i, i+d/2), NOT adjacent pairs (2i, 2i+1).")

# rotate_half applied to the SLICED q_rot makes the 64-dim rotary block behave
# as 32 pairs (i, i+32).  q_pass (dims 64..255) is copied through untouched:
q = torch.randn(1, 4, N, TC["head_dim"], dtype=torch.float32)
k = torch.randn(1, 4, N, TC["head_dim"], dtype=torch.float32)
qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
print("      q_embed[..., 64:] == q[..., 64:] ?", bool(torch.equal(qe[..., rotary_dim:], q[..., rotary_dim:])))
print("      q_embed[..., :64] == q[..., :64] ?", bool(torch.equal(qe[..., :rotary_dim], q[..., :rotary_dim])),
      " (must be False)")

# manual formula for the rotary block, using the slice-halves convention
q_rot = q[..., :rotary_dim]
c = cos[:, :, :rotary_dim].unsqueeze(1)
s = sin[:, :, :rotary_dim].unsqueeze(1)
manual = q_rot * c + rotate_half(q_rot) * s
print("      max|manual - real| on rotary block =", (manual - qe[..., :rotary_dim]).abs().max().item())
print("      where c has shape", tuple(c.shape), "(cos sliced to first 64 dims)")

# pair up explicitly to give the C engine an unambiguous recipe
q0 = q_rot[..., :rotary_dim // 2]
q1 = q_rot[..., rotary_dim // 2:]
cq = cos[0, :, :rotary_dim].reshape(N, 2, rotary_dim // 2) if cos.dim() == 3 else None
print("\n      C recipe for dims 0..63 of q (per head):")
print("        y[i]   = q[i]*cos[i] - q[i+32]*sin[i]     for i in 0..31")
print("        y[i+32]= q[i]*sin[i] + q[i+32]*cos[i]     for i in 0..31")
print("      verify: ", end="")
y = torch.empty_like(q_rot)
y[..., :32] = q0 * cos[0, :, :32].unsqueeze(0).unsqueeze(0) - q1 * sin[0, :, :32].unsqueeze(0).unsqueeze(0)
y[..., 32:] = q0 * sin[0, :, :32].unsqueeze(0).unsqueeze(0) + q1 * cos[0, :, :32].unsqueeze(0).unsqueeze(0)
print("max|y - real| =", (y - qe[..., :rotary_dim]).abs().max().item())
print("=" * 78)

# ============================ numeric repeat_kv (Q13) ============================
print("\n[Q13] numeric proof of the head->kv mapping")
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

nkv, groups, head_dim, slen = 2, 4, 4, 1
kv = torch.zeros(1, nkv, slen, head_dim)
for h in range(nkv):
    kv[0, h, 0, :] = h + 1.0            # kv head 0 -> all ones, kv head 1 -> all twos
rep = repeat_kv(kv, groups)
print("      repeat_kv(kv where kvhead0=1, kvhead1=2, groups=4).reshape(-1)[per head] =",
      rep[0, :, 0, 0].tolist())
print("      => head h uses kv head  floor(h / groups) =", [h // groups for h in range(nkv * groups)])
print("      (NOT h % nkv  ->", [h % nkv for h in range(nkv * groups)], ")")

# ============================ numeric apply_rotary slicing ======================
print("\n[Q11] cos used by apply_rotary_pos_emb (full-tensor case): rotary_dim = cos.shape[-1]")
print("      cos handed to apply_rotary_pos_emb is the WHOLE tensor of last-dim", cos.shape[-1])
print("      -> rotary_dim = 64, q_rot = q[..., :64], q_pass = q[..., 64:256], so 192 dims pass through")
print("      (In exp_rotary's earlier calls I sliced cos to 64 manually; both give the same result")
print("       because cos[..., :64] == cos[..., 64:] for this duplicated layout.)")
