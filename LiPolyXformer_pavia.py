"""
LiPolyXformer_pavia.py

PAVIA runner with normalized, learnable depthwise 1D conv coefficient smoother.
This keeps the 1D conv (for novelty) but constrains it to act like an LPF at runtime:
  - softplus(self.coef_smooth.weight) -> positive
  - divide by sum over kernel -> sum-to-1 per channel

Usage:
    python LiPolyXformer_pavia.py --mat pavia_split.mat --epochs 20 --lr 0.005
"""


import os
import time
import random
from typing import Tuple

import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############################################
# Utilities
############################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


############################################
# Losses: Gaussian NLL (uncertainty-aware) + SAM + regularizers
############################################
class HsiUncLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood reconstruction loss with per-band variance (aleatoric uncertainty),
    plus SAM term and regularization on mixer coefficients (TV) and abundance sparsity.
    """
    def __init__(self, beta_sam: float = 0.25, lambda_tv: float = 1e-3, lambda_l1z: float = 1e-4, eps: float = 1e-8):
        super().__init__()
        self.beta_sam = beta_sam
        self.lambda_tv = lambda_tv
        self.lambda_l1z = lambda_l1z
        self.eps = eps

    def forward(self, x_hat: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor,
                coef_smooth: torch.Tensor, abund: torch.Tensor) -> torch.Tensor:
        # x_hat, target, logvar: [tokens, Bands]
        var = torch.exp(torch.clamp(logvar, min=-7.0, max=7.0))  # numerical stability
        nll = 0.5 * ((target - x_hat) ** 2 / (var + 1e-8) + torch.log(var + 1e-8))
        nll = nll.mean()
        # SAM (mean spectral angle)
        num = (x_hat * target).sum(dim=1)
        den = x_hat.norm(dim=1) * target.norm(dim=1) + self.eps
        cos = torch.clamp(num / den, -1.0 + 1e-7, 1.0 - 1e-7)
        sam = torch.acos(cos).mean()
        # TV on mixer coefficients across bands (coef_smooth: [tokens, Bands, K])
        tv = (coef_smooth[:, 1:, :] - coef_smooth[:, :-1, :]).abs().mean()
        # L1 on abundances for sparsity (abund: [tokens, E])
        l1z = abund.abs().mean()
        return nll + self.beta_sam * sam + self.lambda_tv * tv + self.lambda_l1z * l1z


############################################
# Core blocks
############################################
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_hid, d_out), nn.Dropout(p)
        )
    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d_model: int, heads: int = 6, p: float = 0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.h = heads
        self.d = d_model // heads
        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(p)
        self.scale = self.d ** -0.5
    def forward(self, x: torch.Tensor):
        # x: [N, D]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(x.size(0), self.h, self.d).permute(1, 0, 2)  # [h,N,d]
        k = k.view(x.size(0), self.h, self.d).permute(1, 0, 2)
        v = v.view(x.size(0), self.h, self.d).permute(1, 0, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(x.size(0), -1)
        out = self.drop(self.proj(out))
        return out


class CrossAttention(nn.Module):
    """Queries from x attend to keys/values from y, with auto-sized inner dim."""
    def __init__(self, d_q: int, d_kv: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        assert heads >= 1
        self.h = heads
        # pick a shared inner size that's the next multiple of heads
        d_base = min(d_q, d_kv)
        d_inner = ((d_base + heads - 1) // heads) * heads  # round up to multiple of heads
        self.dh = d_inner // heads

        self.Wq = nn.Linear(d_q, d_inner, bias=True)
        self.Wk = nn.Linear(d_kv, d_inner, bias=True)
        self.Wv = nn.Linear(d_kv, d_inner, bias=True)
        self.proj = nn.Linear(d_inner, d_q, bias=True)
        self.drop = nn.Dropout(p)
        self.scale = self.dh ** -0.5

    def forward(self, xq: torch.Tensor, ykv: torch.Tensor):
        # xq: [Nq, d_q], ykv: [Nk, d_kv]
        Nq, _ = xq.shape
        Nk, _ = ykv.shape

        q = self.Wq(xq).view(Nq, self.h, self.dh).permute(1, 0, 2)   # [h, Nq, dh]
        k = self.Wk(ykv).view(Nk, self.h, self.dh).permute(1, 0, 2)  # [h, Nk, dh]
        v = self.Wv(ykv).view(Nk, self.h, self.dh).permute(1, 0, 2)  # [h, Nk, dh]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale     # [h, Nq, Nk]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(Nq, -1)  # [Nq, h*dh]
        out = self.drop(self.proj(out))  # back to d_q
        return out



############################################
# LiPolyXformer Model: Spectral–Spatial Cross-Attention + Learnable Polynomial Mixer + Uncertainty
############################################
class LiPolyXformer(nn.Module):
    """
    - Spatial MHSA over tokens (10x10 = 100 tokens, feature=Bands)
    - Spectral MHSA over bands (sequence=Bands, feature=tokens)
    - Cross-attention fusion (spatial<->spectral)
    - Abundance bottleneck (softmax)
    - Learnable polynomial mixer: x_hat = p_l + a2*p_l^2 + a3*p_l^3 (coeffs are per-band, smoothed, bounded via sigmoid)
    - Uncertainty head: per-band log-variance for Gaussian NLL
    """
    def __init__(self,
                 Bands: int,
                 num_endmembers: int = 10,
                 heads_spatial: int = 6,
                 heads_spectral: int = 4,
                 heads_cross: int = 4,
                 p: float = 0.1,
                 coef_kernel: int = 7,
                 coef_learnable: bool = True):
        super().__init__()
        self.Bands = Bands
        self.tokens = 100

        # Positional embedding for tokens
        self.pos_tokens = nn.Parameter(torch.empty(self.tokens, Bands))
        nn.init.normal_(self.pos_tokens, std=0.02)

        # Spatial branch
        self.ln_spa1 = nn.LayerNorm(Bands)
        self.attn_spa = MHSA(Bands, heads_spatial, p)
        self.ln_spa2 = nn.LayerNorm(Bands)
        self.mlp_spa = MLP(Bands, num_endmembers, Bands, p)

        # Spectral branch (bands as sequence; feature=tokens)
        self.ln_spec1 = nn.LayerNorm(self.tokens)
        self.attn_spec = MHSA(self.tokens, heads_spectral, p)
        self.ln_spec2 = nn.LayerNorm(self.tokens)
        self.mlp_spec = MLP(self.tokens, self.tokens, self.tokens, p)

        # Cross Attention (two directions)
        self.cross_spa_from_spec = CrossAttention(d_q=Bands, d_kv=self.tokens, heads=heads_cross, p=p)
        self.cross_spec_from_spa = CrossAttention(d_q=self.tokens, d_kv=Bands, heads=heads_cross, p=p)

        # Fusion gates
        self.gate_spa = nn.Sequential(nn.Linear(Bands, Bands), nn.Sigmoid())
        self.gate_spec = nn.Sequential(nn.Linear(self.tokens, self.tokens), nn.Sigmoid())

        # Abundance head
        self.to_abund = nn.Linear(Bands, num_endmembers, bias=False)
        nn.init.xavier_normal_(self.to_abund.weight)
        self.sm = nn.Softmax(dim=1)

        # Linear reconstruction from abundances
        self.to_spec = nn.Linear(num_endmembers, Bands, bias=False)
        nn.init.xavier_normal_(self.to_spec.weight)

        # Mixer coefficients predictor a2, a3 from [p_l, x*p_l]
        self.coef_pred = nn.Sequential(
            nn.Linear(2 * Bands, 2 * Bands), nn.GELU(),
            nn.Linear(2 * Bands, 2 * Bands), nn.GELU(),
            nn.Linear(2 * Bands, 2 * Bands)  # outputs concatenated [a2, a3]
        )

        # -------------------------
        # Coefficient smoother: depthwise 1D conv (learnable optional)
        # -------------------------
        k = int(coef_kernel)
        if k % 2 == 0:
            k += 1
        pad = (k - 1) // 2

        # depthwise conv: in_channels=2 (a2,a3), groups=2 => per-channel conv
        self.coef_smooth = nn.Conv1d(in_channels=2, out_channels=2,
                                     kernel_size=k, padding=pad,
                                     groups=2, bias=True)

        # initialize to average LPF as starting point
        # initialize to average LPF as starting point
        with torch.no_grad():
            lp = torch.ones((k,), dtype=torch.float32) / float(k)
            for ch in range(2):
                self.coef_smooth.weight[ch, 0, :] = torch.tensor(lp)
            if self.coef_smooth.bias is not None:
                self.coef_smooth.bias.zero_()


        # freeze smoother if requested (non-learnable LPF)
        if not bool(coef_learnable):
            for p in self.coef_smooth.parameters():
                p.requires_grad = False

        # Uncertainty head: per-band log-variance
        self.logvar_head = nn.Sequential(
            nn.Linear(Bands, Bands), nn.GELU(), nn.Linear(Bands, Bands)
        )

    def forward(self, x: torch.Tensor):
        # x: [tokens=100, Bands]
        tokens, B = x.shape
        assert tokens == self.tokens and B == self.Bands

        # Spatial branch
        xs = x + self.pos_tokens
        xs = self.ln_spa1(xs)
        xs = xs + self.attn_spa(xs)
        xs = self.ln_spa2(xs)
        xs = xs + self.mlp_spa(xs)

        # Spectral branch
        xb = x.transpose(0, 1)  # [Bands, tokens]
        xb = self.ln_spec1(xb)
        xb = xb + self.attn_spec(xb)
        xb = self.ln_spec2(xb)
        xb = xb + self.mlp_spec(xb)

        # Cross-attention
        xs_cross = xs + self.cross_spa_from_spec(xs, xb)  # [tokens,Bands]
        xb_cross = xb + self.cross_spec_from_spa(xb, xs)  # [Bands, tokens]

        # Gates
        xs = xs + self.gate_spa(xs) * xs_cross
        xb = xb + self.gate_spec(xb) * xb_cross
        xb = xb.transpose(0, 1)  # [tokens, Bands]

        # Fuse
        xf = xs + xb

        # Abundances and linear part
        z = self.sm(self.to_abund(xf))           # [tokens, E]
        p_l = self.to_spec(z)                    # [tokens, Bands]

        # Mixer coefficients (per-band a2, a3), smoothed & bounded to [0,1]
        xy = x * p_l
        coef_raw = self.coef_pred(torch.cat([p_l, xy], dim=1))  # [tokens, 2*Bands]
        a2_raw, a3_raw = coef_raw[:, :B], coef_raw[:, B:]
        coef_stack = torch.stack([a2_raw, a3_raw], dim=1)  # [tokens, 2, Bands]
        coef_sm = self.coef_smooth(coef_stack)             # [tokens, 2, Bands]
        a2 = torch.sigmoid(coef_sm[:, 0, :])
        a3 = torch.sigmoid(coef_sm[:, 1, :])

        # Polynomial mixer up to order 3
        x_hat = p_l + a2 * (p_l ** 2) + a3 * (p_l ** 3)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        # Uncertainty (log-variance per band)
        logvar = self.logvar_head(xf)

        # For regularization TV, return smoothed coef tensor as [tokens, Bands, K]
        coef_for_tv = torch.stack([a2, a3], dim=-1)
        return z, p_l, x_hat, logvar, coef_for_tv




############################################
# Data IO
############################################

def load_pavia_split(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(mat_path, 'r') as f:
        groundtruth = np.transpose(f['groundtruth'][:])
        data_train = np.transpose(f['data_train'][:])
        data_full = np.transpose(f['data'][:])
        data_test = np.transpose(f['data_test'][:])
    return data_train, groundtruth, data_full, data_test


############################################
# Train & Test
############################################

'''def train(model: nn.Module, data_train: np.ndarray, lr: float = 5e-3, epochs: int = 20, device: str = None,
          beta_sam: float = 0.25, lambda_tv: float = 1e-3, lambda_l1z: float = 1e-4):'''
def train(model: nn.Module, data_train: np.ndarray, lr: float = 5e-3, epochs: int = 20, device: str = None,
          beta_sam: float = 0.25, lambda_tv: float = 1e-3, lambda_l1z: float = 1e-4, use_mse: bool = False):
      
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    '''optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = HsiUncLoss(beta_sam=beta_sam, lambda_tv=lambda_tv, lambda_l1z=lambda_l1z)
    '''
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_mse:
        criterion_mse = nn.MSELoss()
        # wrapper to match existing call signature used in loop
        def _mse_wrapper(x_hat, target, logvar, coef_tv, z):
            return criterion_mse(x_hat, target)
        criterion = _mse_wrapper
    else:
        criterion = HsiUncLoss(beta_sam=beta_sam, lambda_tv=lambda_tv, lambda_l1z=lambda_l1z)

    data = torch.from_numpy(data_train).float().to(device)  # [361, 100, Bands]
    num_windows = data.shape[0]

    loss_hist = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        indices = list(range(num_windows))
        random.shuffle(indices)
        epoch_loss = 0.0

        for i in range(19):
            optimizer.zero_grad()
            batch_loss = 0.0
            for j in range(19):
                idx = indices[i * 19 + j]
                x = data[idx]
                z, p_l, x_hat, logvar, coef_tv = model(x)
                loss = criterion(x_hat, x, logvar, coef_tv, z)
                loss.backward()
                batch_loss += float(loss.item())
                epoch_loss += float(loss.item())
            optimizer.step()
        t1 = time.perf_counter() - t0
        loss_hist.append(epoch_loss)
        print(f"[epoch {epoch+1:02d}] loss_batch:{batch_loss:.4f} loss_epoch:{epoch_loss:.4f} time:{t1:.2f}s")

    return model, loss_hist


def test(model: nn.Module, data_test: np.ndarray, data_full: np.ndarray, device: str = None, use_uncertainty_blend: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data_t = torch.from_numpy(data_test).float().to(device)  # [361, 100, Bands]
    Bands = data_t.shape[-1]

    '''result = torch.zeros(100, 100, Bands, device=device)
    weight = torch.zeros(100, 100, 1, device=device)'''
    result = torch.zeros(100, 100, Bands, device=device)
    weight = torch.zeros(100, 100, 1, device=device)

    # for saving per-pixel mean variance map (τ): accumulate mean variance like result & weight
    var_result = torch.zeros(100, 100, device=device)
    var_weight = torch.zeros(100, 100, device=device)


    with torch.no_grad():
        for i in range(361):
            x = data_t[i]
            z, p_l, x_hat, logvar, coef_tv = model(x)
            patch = x_hat.view(10, 10, Bands).transpose(0, 1)
            row = i // 19
            col = i % 19
            r0, c0 = 5 * row, 5 * col
            '''if use_uncertainty_blend:
                # compute per-pixel weight = 1 / (epsilon + mean variance)
                var = torch.exp(torch.clamp(logvar, min=-7.0, max=7.0))  # [tokens,B]
                var_patch = var.mean(dim=1).view(10, 10).transpose(0, 1)  # [10,10]
                w = 1.0 / (1e-6 + var_patch)
                result[r0:r0+10, c0:c0+10] += patch * w.unsqueeze(-1)
                weight[r0:r0+10, c0:c0+10] += w.unsqueeze(-1)
            else:
                result[r0:r0+10, c0:c0+10] += patch
                weight[r0:r0+10, c0:c0+10] += 1'''
            if use_uncertainty_blend:
                # compute per-pixel weight = 1 / (epsilon + mean variance)
                var = torch.exp(torch.clamp(logvar, min=-7.0, max=7.0))  # [tokens,B]
                var_patch = var.mean(dim=1).view(10, 10).transpose(0, 1)  # [10,10]
                w = 1.0 / (1e-6 + var_patch)
                result[r0:r0+10, c0:c0+10] += patch * w.unsqueeze(-1)
                weight[r0:r0+10, c0:c0+10] += w.unsqueeze(-1)
                # accumulate var (weighted)
                var_result[r0:r0+10, c0:c0+10] += var_patch * w
                var_weight[r0:r0+10, c0:c0+10] += w
            else:
                result[r0:r0+10, c0:c0+10] += patch
                weight[r0:r0+10, c0:c0+10] += 1
                # accumulate var without weights (treat weight=1)
                var = torch.exp(torch.clamp(logvar, min=-7.0, max=7.0))
                var_patch = var.mean(dim=1).view(10, 10).transpose(0, 1)
                var_result[r0:r0+10, c0:c0+10] += var_patch
                var_weight[r0:r0+10, c0:c0+10] += 1


    result = result / torch.clamp(weight, min=1.0)
    # compute per-pixel mean variance map (τ)
    var_map = (var_result / torch.clamp(var_weight, min=1.0)).detach().cpu().numpy()  # shape [100,100]

    data_full_t = torch.from_numpy(data_full).float().to(device)
    err = (result - data_full_t).pow(2).sum(dim=2).sqrt()
    err_np = err.detach().cpu().numpy()
    e_min, e_max = float(err_np.min()), float(err_np.max())
    err_norm = (err_np - e_min) / (e_max - e_min + 1e-8)

    return err_norm, result.detach().cpu().numpy(), var_map


    '''data_full_t = torch.from_numpy(data_full).float().to(device)
    err = (result - data_full_t).pow(2).sum(dim=2).sqrt()
    err_np = err.detach().cpu().numpy()
    e_min, e_max = float(err_np.min()), float(err_np.max())
    err_norm = (err_np - e_min) / (e_max - e_min + 1e-8)

    return err_norm, result.detach().cpu().numpy()'''


############################################
# Main
############################################
if __name__ == "__main__":
    import argparse
    import sys
    import types

    parser = argparse.ArgumentParser(description="LiPolyXformer on Pavia: Cross-Attn + Polynomial Mixer + Uncertainty")

    # dataset / training
    parser.add_argument("--mat", type=str, default="pavia_split.mat")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--heads_spatial", type=int, default=6)
    parser.add_argument("--heads_spectral", type=int, default=4)
    parser.add_argument("--heads_cross", type=int, default=4)
    parser.add_argument("--beta_sam", type=float, default=0.25)
    parser.add_argument("--lambda_tv", type=float, default=1e-3)
    parser.add_argument("--lambda_l1z", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_unc_blend", action='store_true', help="Disable uncertainty-weighted overlap blending")

    # coef_smoother controls
    parser.add_argument("--coef-kernel", type=int, default=7,
                        help="Kernel size for depthwise coefficient smoother (odd number, e.g. 5,7,9).")
    parser.add_argument("--no-coef-learnable", action="store_true",
                        help="If passed, freeze coefficient smoother (non-learnable). By default it is learnable.")

    # Ablation toggles (safe runtime monkeypatches; no architecture edit)
    parser.add_argument("--num_endmembers", type=int, default=10,
                        help="Number of pseudo-endmembers (abundance dim).")
    parser.add_argument("--no_poly", action="store_true",
                        help="Disable polynomial mixer (force a2=a3≈0).")
    parser.add_argument("--no_cross", action="store_true",
                        help="Disable both cross-attention modules (spa<->spec).")
    parser.add_argument("--one_way", type=str, default=None, choices=["spa_from_spec","spec_from_spa", "none"],
                        help="Enable one-way cross-attention (spa_from_spec or spec_from_spa).")
    parser.add_argument("--no_gates", action="store_true",
                        help="Disable fusion gating (set gates to 0 so cross contributions are ignored).")
    parser.add_argument("--attn_spa_off", action="store_true",
                        help="Disable spatial MHSA contribution (attn_spa -> zero).")
    parser.add_argument("--attn_spec_off", action="store_true",
                        help="Disable spectral MHSA contribution (attn_spec -> zero).")
    parser.add_argument("--no_uncertainty", action="store_true",
                        help="Disable uncertainty head (stop predicting varying logvar; returns zeros).")
    parser.add_argument("--use_mse", action="store_true",
                        help="Use MSE loss instead of Gaussian NLL (keeps uncertainty head but not used).")

    # output directory for this run (so you can run many experiments independently)
    parser.add_argument("--out_dir", type=str, default="outputs_pavia_novel",
                        help="Directory to save results for this run.")

    args = parser.parse_args()

    # interpret flags
    coef_kernel = int(args.coef_kernel)
    coef_learnable = not bool(args.no_coef_learnable)

    # ensure out_dir exists
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)

    if not os.path.exists(args.mat):
        raise FileNotFoundError(f"{args.mat} not found. Generate it via preprocessing (win=10, step=5).")

    data_train, gt, data_full, data_test = load_pavia_split(args.mat)
    Bands = data_train.shape[-1]

    # instantiate model; pass the new args into constructor
    model = LiPolyXformer(Bands, num_endmembers=args.num_endmembers,
                  heads_spatial=args.heads_spatial,
                  heads_spectral=args.heads_spectral,
                  heads_cross=args.heads_cross,
                  p=0.1,
                  coef_kernel=coef_kernel,
                  coef_learnable=coef_learnable)

    # -------------------- runtime ablation monkeypatches --------------------
    # helper: zero forward for cross/attn modules (returns zeros shaped like queries)
    def _zero_forward(self, *args, **kwargs):
        q = args[0]
        return torch.zeros_like(q)

    # 1) disable both cross-attn
    if args.no_cross:
        model.cross_spa_from_spec.forward = types.MethodType(_zero_forward, model.cross_spa_from_spec)
        model.cross_spec_from_spa.forward = types.MethodType(_zero_forward, model.cross_spec_from_spa)

    # 2) one-way cross-attention
    if args.one_way == "spa_from_spec":
        # keep spa<-spec, disable spec<-spa
        model.cross_spec_from_spa.forward = types.MethodType(_zero_forward, model.cross_spec_from_spa)
    elif args.one_way == "spec_from_spa":
        model.cross_spa_from_spec.forward = types.MethodType(_zero_forward, model.cross_spa_from_spec)

    # 3) disable polynomial mixer (force a2,a3 ~ 0)
    if args.no_poly:
        def _coef_zero_forward(self, x):
            B = model.Bands
            return -20.0 * torch.ones((x.size(0), 2 * B), device=x.device)
        model.coef_pred.forward = types.MethodType(_coef_zero_forward, model.coef_pred)

    # 4) disable gates (set gate output to zeros so cross contributions are ignored)
    if args.no_gates:
        def _gate_zero(self, x):
            return torch.zeros_like(x)
        model.gate_spa.forward = types.MethodType(_gate_zero, model.gate_spa)
        model.gate_spec.forward = types.MethodType(_gate_zero, model.gate_spec)

    # 5) disable whole attention branches (spatial / spectral) if requested
    if args.attn_spa_off:
        model.attn_spa.forward = types.MethodType(_zero_forward, model.attn_spa)
    if args.attn_spec_off:
        model.attn_spec.forward = types.MethodType(_zero_forward, model.attn_spec)

    # 6) disable uncertainty head (make it output zeros -> var = exp(0)=1 constant)
    if args.no_uncertainty:
        def _logvar_zero_forward(self, x):
            return torch.zeros_like(x)
        model.logvar_head.forward = types.MethodType(_logvar_zero_forward, model.logvar_head)
    # -----------------------------------------------------------------------

    # TRAIN + TEST
    t0 = time.perf_counter()
    model, loss_hist = train(model, data_train, lr=args.lr, epochs=args.epochs,
                             beta_sam=args.beta_sam, lambda_tv=args.lambda_tv, lambda_l1z=args.lambda_l1z,
                             use_mse=args.use_mse)
    # test: note test() now returns (err_norm, recon, var_map)
    score_map, recon, var_map = test(model, data_test, data_full, use_uncertainty_blend=not args.no_unc_blend)
    elapsed = time.perf_counter() - t0

    # Evaluate
    gt1d = gt.reshape(-1).astype(np.int32)
    scr1d = score_map.reshape(-1)
    fpr, tpr, thr = roc_curve(gt1d, scr1d)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f} | Total time: {elapsed:.2f}s")

    # Save results to out_dir
    np.save(os.path.join(out_dir, "anomaly_map.npy"), score_map)
    np.save(os.path.join(out_dir, "reconstruction.npy"), recon)
    np.save(os.path.join(out_dir, "loss_hist.npy"), np.array(loss_hist))
    np.save(os.path.join(out_dir, "var_map.npy"), var_map)
    # save flattened gt and scores too for convenience
    np.save(os.path.join(out_dir, "gt_flat.npy"), gt1d)
    np.save(os.path.join(out_dir, "score_flat.npy"), scr1d)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"AUC: {roc_auc:.6f}\n")
        f.write(f"Time: {elapsed:.2f}s\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")

    # Optional visualization if available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.title("Anomaly score (0-1)")
        plt.imshow(score_map, cmap="viridis")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title(f"ROC (AUC={roc_auc:.3f})")
        plt.plot(fpr, tpr)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.tight_layout(); plt.show()
    except Exception:
        pass
