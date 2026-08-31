import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn


# ───────── utilities ─────────
def _dip(x, dim=-1, eps=1e-8):
    return torch.log(torch.mean(x.pow(2), dim=dim) + eps)


def normalize_adjacency(A):
    deg = A.sum(dim=1).clamp(min=1e-6); d = deg.pow(-0.5)
    return d.unsqueeze(1) * A * d.unsqueeze(0)


def _shift_time(x, lag):
    if lag == 0:
        return x
    T = x.shape[-1]; out = torch.zeros_like(x)
    if lag > 0:
        out[..., :T - lag] = x[..., lag:]
    else:
        out[..., -lag:] = x[..., :T + lag]
    return out


def compute_k_adjacency(A, k):
    """disentangled exact-hop k-adjacency (self-loop 포함). k=0 → I."""
    N = A.shape[0]
    if k == 0:
        return torch.eye(N, device=A.device)
    At = A.clone(); At.fill_diagonal_(1)
    P = torch.eye(N, device=A.device); Pprev = None
    for s in range(k):
        if s == k - 1:
            Pprev = P.clone()
        P = P @ At
    Rk = (P > 0).float(); Rkm1 = (Pprev > 0).float()
    return ((Rk - Rkm1 + torch.eye(N, device=A.device)) > 0).float()


class WPLIAdjacencyBuilder:
    """train-set WPLI → 채널별 top-k(16) 이웃 binary mask (대칭)."""

    @staticmethod
    def build(X_train_raw, top_k=16):
        N_s, C, T = X_train_raw.shape
        print(f"  WPLI: {N_s} trials, {C}ch, {T}smp...", end=" ", flush=True)
        an = signal.hilbert(X_train_raw, axis=-1).astype(np.complex64)
        num = np.zeros((C, C), np.float64); den = np.zeros((C, C), np.float64)
        for i in range(N_s):
            cr = an[i][:, None, :] * np.conj(an[i][None, :, :]); im = np.imag(cr)
            num += im.sum(-1).astype(np.float64); den += np.abs(im).sum(-1).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.where(den > 1e-10, np.abs(num) / den, 0.0).astype(np.float32)
        np.fill_diagonal(w, 0.0)
        k = min(top_k, C - 1); idx = np.argpartition(w, -k, axis=1)[:, -k:]
        mask = np.zeros((C, C), np.float32); mask[np.arange(C)[:, None], idx] = 1.0
        mask = np.maximum(mask, mask.T); np.fill_diagonal(mask, 0.0)
        print(f"done (top_k={k}, edges={int(mask.sum())//2})", flush=True)
        return torch.tensor(mask, dtype=torch.float32)


class SimAM1d(nn.Module):
    """parameter-free 1-D(시간축) SimAM."""

    def __init__(self, lambda_=1e-4):
        super().__init__(); self.lambda_ = lambda_

    def forward(self, x):
        n = x.shape[2] - 1
        if n <= 0:
            return x
        d = (x - x.mean(dim=2, keepdim=True)).pow(2)
        v = d.sum(dim=2, keepdim=True) / n
        return x * torch.sigmoid(d / (4 * (v + self.lambda_)) + 0.5)


# ───────── proposed aggregation (통합 residual + 0~2hop + post-sum SimAM) ─────────
class ProposedAggregation(nn.Module):
    r"""
    Y_t = BN( SimAM( Σ_{k=0}^{2} S_k · ( Σ_{δ∈Δ} X_{t+δ} ) ) ) → ELU

      S_k = Â_k + M^{(k)} ⊙ R_k        (k∈{0,1,2}; 0-hop=대각 self, R_k 통합=모든 lag 공유)
    scale(lag)·hop 모두 단순 add, SimAM 은 hop 합 후 1회(post-sum).
    """

    def __init__(self, num_nodes=64, num_scales=3, lags=(-1, 0, 1), res_init_scale=1e-1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hops = list(range(num_scales))          # [0,1,2] — 0-hop 유지
        self.lags = list(lags)
        for k in self.hops:
            self.register_buffer(f"A_k_{k}", torch.eye(num_nodes))
            self.register_buffer(f"A_mask_{k}", torch.eye(num_nodes))
        # 통합(shared-lag) residual: hop당 단일 N×N, 모든 lag 공유
        self.R = nn.Parameter(torch.empty(len(self.hops), num_nodes, num_nodes)
                              .uniform_(-res_init_scale, res_init_scale))
        self.simam = SimAM1d()                        # post-sum SimAM
        self.bn = nn.BatchNorm1d(num_nodes)
        self.act = nn.ELU()

    @torch.no_grad()
    def init_adj(self, A_binary):
        for k in self.hops:
            A_k = compute_k_adjacency(A_binary, k)
            getattr(self, f"A_k_{k}").copy_(normalize_adjacency(A_k))
            getattr(self, f"A_mask_{k}").copy_((A_k != 0).float())

    def masked_residuals(self):
        return [self.R[i] * getattr(self, f"A_mask_{k}") for i, k in enumerate(self.hops)]

    def forward(self, x):
        xbar = sum(_shift_time(x, lag) for lag in self.lags)     # Σ_δ X_{t+δ} (lag add)
        out = 0
        for i, k in enumerate(self.hops):
            S = getattr(self, f"A_k_{k}") + getattr(self, f"A_mask_{k}") * self.R[i]
            out = out + torch.einsum("ij,bjt->bit", S, xbar)      # hop add
        out = self.simam(out)                                     # SimAM(Σ_k agg_k)
        return self.act(self.bn(out))


# ───────── lightweight MS-TCL ─────────
class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch=512, kernel_sizes=(3, 5)):
        super().__init__()
        branch_ch = max(out_ch // len(kernel_sizes), 16)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, 1), nn.BatchNorm1d(branch_ch), nn.ELU(),
                nn.Conv1d(branch_ch, branch_ch, k, padding=pad, groups=branch_ch),
                nn.BatchNorm1d(branch_ch), nn.ELU()))
        self.proj = nn.Sequential(nn.Conv1d(branch_ch * len(kernel_sizes), out_ch, 1), nn.BatchNorm1d(out_ch))
        self.simam = SimAM1d()
        self.residual = nn.Conv1d(in_ch, out_ch, 1)
        self.elu = nn.ELU()
        self.downsample = nn.AvgPool1d(2, 2)

    def forward(self, x):
        res = self.residual(x)
        out = self.simam(self.proj(torch.cat([b(x) for b in self.branches], 1)))
        return self.downsample(self.elu(out + res))


# ───────── main model ─────────
class ProposedFinal(nn.Module):
    def __init__(self, num_nodes=64, num_classes=5, num_scales=3,
                 branch_a_lags=(-1, 0, 1), branch_b_lags=(-4, -2, 0, 2, 4),
                 temp_out_ch=512, temp_kernel_sizes=(3, 5)):
        super().__init__()
        self.init_pool = nn.AvgPool1d(2, 2)
        self.agg_a = ProposedAggregation(num_nodes, num_scales, branch_a_lags)
        self.agg_b = ProposedAggregation(num_nodes, num_scales, branch_b_lags)
        self.temporal_conv = MultiScaleTemporalConv(2 * num_nodes, temp_out_ch, temp_kernel_sizes)
        self.temporal_attn = nn.Linear(temp_out_ch, 1, bias=False)
        self.classifier = nn.Linear(2 * temp_out_ch, num_classes)

    def init_adjacency(self, X_train_raw):
        A = WPLIAdjacencyBuilder.build(X_train_raw)
        self.agg_a.init_adj(A); self.agg_b.init_adj(A)

    def sparsity_l1(self):
        tot = 0.0
        for br in (self.agg_a, self.agg_b):
            for r in br.masked_residuals():
                tot = tot + r.abs().sum()
        return tot

    def forward(self, x):
        x = self.init_pool(x)
        x = torch.cat([self.agg_a(x), self.agg_b(x)], dim=1)
        x = self.temporal_conv(x)
        feat_dip = _dip(x, dim=-1)
        attn = torch.softmax(self.temporal_attn(x.permute(0, 2, 1)), dim=1)
        feat_attn = torch.bmm(x, attn).squeeze(-1)
        return self.classifier(torch.cat([feat_dip, feat_attn], dim=-1))
