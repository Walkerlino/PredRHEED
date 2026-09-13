from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn.functional as F


CHUNK = 2048
SP_THRESHOLD = 4096


class ChunkedVAttnT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, scale, chunk=CHUNK):
        B, Sq, dk = Q.shape
        C = V.shape[1]
        Z = torch.empty(B, C, Sq, device=Q.device, dtype=Q.dtype)
        for s in range(0, Sq, chunk):
            e = min(Sq, s + chunk)
            A = torch.softmax(torch.bmm(Q[:, s:e], K) / scale, dim=-1)
            Z[:, :, s:e] = torch.bmm(V, A.transpose(1, 2))
        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale
        ctx.chunk = chunk
        return Z

    @staticmethod
    def backward(ctx, dZ):
        Q, K, V = ctx.saved_tensors
        scale, chunk = ctx.scale, ctx.chunk
        B, Sq, dk = Q.shape
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dZ = dZ.contiguous()
        for s in range(0, Sq, chunk):
            e = min(Sq, s + chunk)
            A = torch.softmax(torch.bmm(Q[:, s:e], K) / scale, dim=-1)
            dZc = dZ[:, :, s:e]
            dV += torch.bmm(dZc, A)
            dA = torch.bmm(dZc.transpose(1, 2), V)
            dS = A * (dA - (dA * A).sum(dim=-1, keepdim=True))
            dQ[:, s:e] = torch.bmm(dS, K.transpose(1, 2)) / scale
            dK += torch.bmm(Q[:, s:e].transpose(1, 2), dS) / scale
        return dQ, dK, dV, None, None


def sam_forward_maybe_chunked(
    self, h_t: torch.Tensor, m_prev: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    B, C, H, W = h_t.shape
    sr = self.spatial_reduction
    h_small = F.adaptive_avg_pool2d(h_t, (H // sr, W // sr))
    m_small = F.adaptive_avg_pool2d(m_prev, (H // sr, W // sr))
    _, _, hs, ws = h_small.shape
    sp = hs * ws
    scale = np.sqrt(self.d_k)

    Q_h = self.W_hq(h_small).view(B, self.d_k, sp).permute(0, 2, 1)
    K_h = self.W_hk(h_small).view(B, self.d_k, sp)
    K_m = self.W_mk(m_small).view(B, self.d_k, sp)
    V_h = self.W_hv(h_small).view(B, C, sp)
    V_m = self.W_mv(m_small).view(B, C, sp)

    if sp > SP_THRESHOLD:
        Z_h = ChunkedVAttnT.apply(Q_h, K_h, V_h, float(scale), CHUNK).view(B, C, hs, ws)
        Z_m = ChunkedVAttnT.apply(Q_h, K_m, V_m, float(scale), CHUNK).view(B, C, hs, ws)
    else:
        A_h = F.softmax(torch.bmm(Q_h, K_h) / scale, dim=-1)
        A_m = F.softmax(torch.bmm(Q_h, K_m) / scale, dim=-1)
        Z_h = torch.bmm(V_h, A_h.permute(0, 2, 1)).view(B, C, hs, ws)
        Z_m = torch.bmm(V_m, A_m.permute(0, 2, 1)).view(B, C, hs, ws)

    Z_h = F.interpolate(Z_h, size=(H, W), mode="bilinear", align_corners=False)
    Z_m = F.interpolate(Z_m, size=(H, W), mode="bilinear", align_corners=False)

    Z = self.W_z(torch.cat([Z_h, Z_m], dim=1))
    Z = self.attn_dropout(Z)

    return self._update_memory(Z, h_t, m_prev)


def patch_model_chunked(model: torch.nn.Module) -> torch.nn.Module:

    attention_layers = []
    for attribute in ("encoder_sam", "decoder_sam", "encoder_sa", "decoder_sa"):
        attention_layers.extend(getattr(model, attribute, ()))
    for sam in attention_layers:
        sam.forward = types.MethodType(sam_forward_maybe_chunked, sam)
    return model


def run_equivalence_check(
    device: str = "cpu", atol_fwd: float = 1e-6, atol_grad: float = 1e-5
) -> dict:

    torch.manual_seed(0)
    results = {}
    for (B, Sq, dk, C, chunk) in [(2, 57, 8, 16, 16), (1, 300, 32, 128, 128),
                                  (3, 128, 16, 32, 50)]:
        Q = torch.randn(B, Sq, dk, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.randn(B, dk, Sq, device=device, dtype=torch.float64, requires_grad=True)
        V = torch.randn(B, C, Sq, device=device, dtype=torch.float64, requires_grad=True)
        scale = float(dk) ** 0.5

        A = torch.softmax(torch.bmm(Q, K) / scale, dim=-1)
        Z_ref = torch.bmm(V, A.transpose(1, 2))
        gref = torch.autograd.grad(Z_ref.pow(2).sum(), (Q, K, V), retain_graph=False)

        Q2 = Q.detach().clone().requires_grad_(True)
        K2 = K.detach().clone().requires_grad_(True)
        V2 = V.detach().clone().requires_grad_(True)
        Z_chk = ChunkedVAttnT.apply(Q2, K2, V2, scale, chunk)
        gchk = torch.autograd.grad(Z_chk.pow(2).sum(), (Q2, K2, V2), retain_graph=False)

        fwd_err = float((Z_ref - Z_chk).abs().max().detach())
        grad_err = float(max((a - b).abs().max() for a, b in zip(gref, gchk)))
        if not fwd_err < atol_fwd:
            raise AssertionError(f"G8 FAILED fwd: {fwd_err}")
        if not grad_err < atol_grad:
            raise AssertionError(f"G8 FAILED grad: {grad_err}")
        results[f"B{B}_S{Sq}_dk{dk}_C{C}_chunk{chunk}"] = {
            "fwd_max_abs_err": fwd_err, "grad_max_abs_err": grad_err}
    return results


__all__ = [
    "CHUNK",
    "SP_THRESHOLD",
    "ChunkedVAttnT",
    "patch_model_chunked",
    "run_equivalence_check",
    "sam_forward_maybe_chunked",
]
