#!/home/wisedu/laya/bin/python
"""
exp_delta_order.py -- determines the EXACT operation order of the gated delta rule
used by Qwen3.5 (transformers/models/qwen3_5/modeling_qwen3_5.py:
torch_recurrent_gated_delta_rule, lines 438-497; torch_chunk_gated_delta_rule, lines 301-434).

Question (SPEC A4): is the recurrent state decayed BEFORE or AFTER computing kv_mem = S @ k ?

  Order (i)  "post-decay read":  kv_mem = S_old @ k ; delta = beta*(v - kv_mem) ;
                                 S_new = gamma*S_old + k (x) delta  (read with UN-decayed state)
  Order (ii) "pre-decay read" :  S_d = gamma*S_old ; delta = beta*(v - S_d @ k) ;
                                 S_new = S_d + k (x) delta          (read with DECAYED state)

Source (modeling_qwen3_5.py:480-491) reads:

    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t      # <-- (1) decay FIRST
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)   # <-- (2) read DECAYED state
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)  # <-- (3)
                                                                            # output uses POST-update state

This script (a) re-implements both orders by hand and prints their numerical difference, and
(b) calls the REAL reference function and shows which hand-written order it matches bit-for-bit.

Run:  /home/wisedu/laya/bin/python exp_delta_order.py
"""
import sys

import torch

torch.manual_seed(0)
DTYPE = torch.float32
B, T, HK, HV, DK, DV = 1, 6, 2, 2, 8, 8  # small shapes; the rule is shape-agnostic


# --------------------------------------------------------------------------------------
# (A) the real reference implementation, imported verbatim from the installed transformers
# --------------------------------------------------------------------------------------
def import_reference():
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )
    except Exception as exc:  # pragma: no cover
        print(f"!! could not import reference implementation: {exc!r}")
        return None, None
    return torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule


# --------------------------------------------------------------------------------------
# (B) hand-written variants
# --------------------------------------------------------------------------------------
def manual(query, key, value, g, beta, initial_state, order, out_state="post", gamma_mode="exp"):
    """Reference semantics minus l2norm/scaling (those are checked separately).

    order == "pre_decay_read"   -> decay S, then read kv_mem from decayed S   (expected: MATCHES)
    order == "post_decay_read"  -> read kv_mem from old S, then decay+write    (expected: DIFFERS)
    gamma_mode: "exp" (correct), "none" (forget exp), "exp_pos" (sign flipped)
    """
    q, k, v, b, d = (x.transpose(1, 2).to(DTYPE).contiguous() for x in (query, key, value, beta, g))
    # reference always scales q by head_dim**-0.5 (meanwhile l2norm is applied outside, in l2norm())
    S = torch.zeros(B, HV, DK, DV, dtype=DTYPE)
    if initial_state is not None:
        S = initial_state.to(DTYPE).clone()
    out = torch.zeros_like(v)
    for i in range(T):
        k_t, v_t, q_t = k[:, :, i], v[:, :, i], q[:, :, i]
        gamma = d[:, :, i].exp()[..., None, None] if gamma_mode == "exp" else (
            d[:, :, i][..., None, None] if gamma_mode == "none" else (-d[:, :, i]).exp()[..., None, None])
        beta_t = b[:, :, i].unsqueeze(-1)
        S_prev = S
        if order == "pre_decay_read":
            S = S * gamma
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        elif order == "post_decay_read":
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            S = S * gamma + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        else:
            raise ValueError(order)
        read = S if out_state == "post" else S_prev
        out[:, :, i] = (read * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous(), S


def l2norm(x, dim=-1, eps=1e-6):  # modeling_qwen3_5.py:294-298
    return x / (x.pow(2).sum(dim=dim, keepdim=True) + eps).sqrt()


def main():
    torch.manual_seed(0)
    query = torch.randn(B, T, HK, DK) * 0.5
    key = torch.randn(B, T, HK, DK) * 0.5
    value = torch.randn(B, T, HV, DV) * 0.5
    beta = torch.rand(B, T, HV)
    g = -torch.rand(B, T, HV) * 0.5  # decay in log space, <= 0  (docstring line 319-320)
    init = torch.randn(B, HV, DK, DV) * 0.35

    print("=" * 78)
    print("shapes: q,k", tuple(query.shape), " v", tuple(value.shape), " g", tuple(g.shape))
    print("g range: [%.4f, %.4f]  -> exp(g) in (0,1): [%.4f, %.4f]"
          % (g.min(), g.max(), g.exp().min(), g.exp().max()))
    print("=" * 78)

    # ---- part 1: the two hand-written orders, no initial state, plain q/k
    o_pre, s_pre = manual(query, key, value, g, beta, None, "pre_decay_read")
    o_post, s_post = manual(query, key, value, g, beta, None, "post_decay_read")
    d = (o_pre - o_post).abs()
    print("\n[1] hand-written orders, initial_state=None")
    print("    max |o_pre_decay_read - o_post_decay_read| = %.6e" % d.max())
    print("    mean|...|                                 = %.6e" % d.mean())
    print("    max |S_pre  - S_post|                     = %.6e" % (s_pre - s_post).abs().max())
    print("    o_pre[0,0,:4]  =", [round(x, 6) for x in o_pre[0, 0].flatten()[:4].tolist()])
    print("    o_post[0,0,:4] =", [round(x, 6) for x in o_post[0, 0].flatten()[:4].tolist()])
    print("    => the two orders are NOT equivalent (relative diff %.3f%%)"
          % (100 * d.max() / o_pre.abs().max()))

    # ---- part 2: with a nonzero initial state (the decode path)
    o_pre_i, s_pre_i = manual(query, key, value, g, beta, init, "pre_decay_read")
    o_post_i, s_post_i = manual(query, key, value, g, beta, init, "post_decay_read")
    di = (o_pre_i - o_post_i).abs()
    print("\n[2] hand-written orders, initial_state = randn*0.35")
    print("    max |o_pre - o_post| = %.6e   (mean %.6e)" % (di.max(), di.mean()))
    print("    o_pre[0,0,:4]  =", [round(x, 6) for x in o_pre_i[0, 0].flatten()[:4].tolist()])
    print("    o_post[0,0,:4] =", [round(x, 6) for x in o_post_i[0, 0].flatten()[:4].tolist()])

    # ---- part 3: compare against the REAL reference function
    ref_rec, ref_chunk = import_reference()
    if ref_rec is None:
        print("\n[3] reference import FAILED -- cannot cross-check")
        return
    # the reference applies l2norm (use_qk_l2norm_in_kernel=True) and scaling q*sqrt(DK)**-1
    qn, kn = l2norm(query, eps=1e-6), l2norm(key, eps=1e-6)
    qs = qn / (DK ** 0.5)
    o_ref, s_ref = ref_rec(qn, kn, value, g=g, beta=beta, initial_state=init,
                           output_final_state=True, use_qk_l2norm_in_kernel=False)
    # ^ l2norm already applied by hand above, so pass False to avoid double-normalising
    o_pre_n, s_pre_n = manual(qs, kn, value, g, beta, init, "pre_decay_read")
    o_post_n, s_post_n = manual(qs, kn, value, g, beta, init, "post_decay_read")

    print("\n[3] cross-check against the REAL torch_recurrent_gated_delta_rule (modeling_qwen3_5.py:438)")
    print("    max |ref - pre_decay_read | = %.6e" % (o_ref - o_pre_n).abs().max())
    print("    max |ref - post_decay_read| = %.6e" % (o_ref - o_post_n).abs().max())
    print("    max |ref_S - pre_S  |       = %.6e" % (s_ref - s_pre_n).abs().max())
    print("    max |ref_S - post_S |       = %.6e" % (s_ref - s_post_n).abs().max())
    print("    ref[0,0,:4]      =", [round(x, 6) for x in o_ref[0, 0].flatten()[:4].tolist()])
    print("    pre[0,0,:4]      =", [round(x, 6) for x in o_pre_n[0, 0].flatten()[:4].tolist()])
    print("    post[0,0,:4]     =", [round(x, 6) for x in o_post_n[0, 0].flatten()[:4].tolist()])
    match = "pre_decay_read (decay FIRST, then read kv_mem)" if \
        (o_ref - o_pre_n).abs().max() < (o_ref - o_post_n).abs().max() else \
        "post_decay_read (read kv_mem from OLD state)"
    print("    ==> REFERENCE MATCHES: %s" % match)

    # ---- part 4: chunk vs recurrent must agree (chunk path is used for prefill)
    o_chunk, s_chunk = ref_chunk(qn, kn, value, g=g, beta=beta, initial_state=init,
                                 output_final_state=True, use_qk_l2norm_in_kernel=False, chunk_size=4)
    print("\n[4] torch_chunk_gated_delta_rule (chunk_size=4) vs torch_recurrent:"
          " max|diff| = %.6e" % (o_chunk - o_ref).abs().max())

    # ---- part 5: output uses the POST-update state
    o_pre_wrongread, _ = manual(qs, kn, value, g, beta, init, "pre_decay_read", out_state="pre")
    print("\n[5] output read with PRE-update vs POST-update state:"
          " max|diff| = %.6e" % (o_ref - o_pre_wrongread).abs().max())
    print("    => output MUST use the state AFTER the current token's update.")

    # ---- part 6: what if you forget exp() on g, or use the wrong sign
    o_noexp, _ = manual(qs, kn, value, g, beta, init, "pre_decay_read", gamma_mode="none")
    o_sign, _ = manual(qs, kn, value, g, beta, init, "pre_decay_read", gamma_mode="exp_pos")
    print("\n[6] sanity: forget exp (gamma = g) -> max|diff| = %.6e ; flip sign (gamma=exp(+|g|)) -> %.6e"
          % ((o_noexp - o_ref).abs().max(), (o_sign - o_ref).abs().max()))


if __name__ == "__main__":
    main()
