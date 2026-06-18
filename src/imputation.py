import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.imputation.mice import MICEData
from src.constants import *

import warnings
warnings.filterwarnings('ignore')

class _GAINNet(nn.Module):
    """3-layer MLP used for both generator and discriminator."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim),     nn.ReLU(),
            nn.Linear(dim, dim),     nn.Sigmoid(),
        )

    def forward(self, x, aux):
        return self.net(torch.cat([x, aux], dim=1))
    
class _DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, emb_dim=64, proj_dim=64):
        super().__init__()
        self.register_buffer('table', self._build(num_steps, emb_dim), persistent=False)
        self.proj1 = nn.Linear(2 * emb_dim, proj_dim)
        self.proj2 = nn.Linear(proj_dim, proj_dim)

    @staticmethod
    def _build(T, dim):
        steps = torch.arange(T).unsqueeze(1).float()
        freqs = 10.0 ** (torch.arange(dim).float() / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * freqs
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, t):
        x = self.table[t]
        return self.proj2(F.silu(self.proj1(x)))

class _TabCSDINet(nn.Module):
    """Feature-axis transformer denoiser. Each column is a token carrying
    [conditional value, noisy target value, conditional mask]."""
    def __init__(self, dim, num_steps, d_model=64, nheads=4, layers=3):
        super().__init__()
        self.token_proj = nn.Linear(3, d_model)
        self.feature_emb = nn.Parameter(torch.randn(dim, d_model) * 0.02)
        self.diff_emb = _DiffusionEmbedding(num_steps, proj_dim=d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=d_model * 2,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, cond_value, noisy_target, cond_mask, t):
        feats = torch.stack([cond_value, noisy_target, cond_mask], dim=-1)  # (B,K,3)
        h = self.token_proj(feats) + self.feature_emb.unsqueeze(0)
        h = h + self.diff_emb(t).unsqueeze(1)                                # (B,1,d)->broadcast
        h = self.transformer(h)
        return self.out(h).squeeze(-1)   

def impute_simple(data, strategy, value=None):
    simple_imp = SimpleImputer(strategy=strategy, fill_value=value)
    return simple_imp.fit_transform(data)

def impute_knn(data):
    n = len(data)
    k = min(int(np.sqrt(n)), n // 2)
    knn_imp = KNNImputer(n_neighbors=k)
    return knn_imp.fit_transform(data)

def impute_random_forest(data, seed):
    rf = RandomForestRegressor(
        random_state=seed,
        n_jobs=N_JOBS,
        n_estimators=ESTIMATORS,
        max_depth=DEPTH,
    )
    rf_imp = IterativeImputer(
        estimator=rf,
        random_state=seed
    )
    return rf_imp.fit_transform(data)

def impute_mice(data, seed):
    np.random.seed(seed)
    mice_imp = MICEData(data)
    mice_imp.update_all(n_iter=ITERATIONS)
    return mice_imp.data.values

def impute_gain(data, seed, batch_size=128, hint_rate=0.9, alpha=100.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(data, dtype=np.float64).copy()
    n, dim = X.shape

    # Mask: 1 = observed, 0 = missing
    mask = (~np.isnan(X)).astype(np.float64)

    # Min-max normalize columns to [0, 1] (GAIN is scale-sensitive)
    col_min = np.nanmin(X, axis=0)
    col_range = np.nanmax(X, axis=0) - col_min
    col_range[col_range == 0] = 1e-6
    Xn = np.nan_to_num((X - col_min) / col_range, nan=0.0)

    Xn = torch.tensor(Xn, dtype=torch.float32, device=device)
    M  = torch.tensor(mask, dtype=torch.float32, device=device)

    G = _GAINNet(dim).to(device)
    D = _GAINNet(dim).to(device)
    opt_G = torch.optim.Adam(G.parameters())
    opt_D = torch.optim.Adam(D.parameters())
    bce = nn.BCELoss()
    eps = 1e-8
    bs  = min(batch_size, n)
    iterations = int(np.clip(300 * max(1, n // bs), 2000, GAIN_ITERATIONS))

    for _ in range(iterations):
        # Sample once; reuse for both D and G steps
        idx  = torch.randperm(n, device=device)[:bs]
        x, m = Xn[idx], M[idx]
        z    = torch.rand_like(x)          # U(0,1) noise as in the GAIN paper
        x_in = m * x + (1 - m) * z
        b    = (torch.rand_like(x) < hint_rate).float()
        h    = b * m + 0.5 * (1 - b)

        # --- Discriminator step (G runs without grad) ---
        with torch.no_grad():
            g = G(x_in, m)
        x_hat = x * m + g * (1 - m)
        d = D(x_hat, h)
        opt_D.zero_grad()
        bce(d, m).backward()
        opt_D.step()

        # --- Generator step ---
        g     = G(x_in, m)
        x_hat = x * m + g * (1 - m)
        d     = D(x_hat, h)
        g_adv = -torch.mean((1 - m) * torch.log(d + eps))
        g_mse = torch.sum((m * x - m * g) ** 2) / (torch.sum(m) + eps)
        opt_G.zero_grad()
        (g_adv + alpha * g_mse).backward()
        opt_G.step()

    # Final imputation over the whole dataset
    with torch.no_grad():
        z    = torch.rand_like(Xn)
        x_in = M * Xn + (1 - M) * z
        g    = G(x_in, M)
        out  = (M * Xn + (1 - M) * g).cpu().numpy()

    # Denormalize and restore observed values exactly
    out = out * col_range + col_min
    out[mask.astype(bool)] = X[mask.astype(bool)]
    return out

def impute_tabcsdi(data, seed, num_steps=20, epochs=TABCSDI_EPOCHS,
                   batch_size=128, n_samples=1,
                   beta_start=1e-4, beta_end=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(data, dtype=np.float64).copy()
    n, dim = X.shape
    obs = (~np.isnan(X)).astype(np.float64)

    # Standardize per column on observed values (diffusion assumes ~N(0,1))
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    col_std[col_std == 0] = 1.0
    Xn = np.nan_to_num((X - col_mean) / col_std, nan=0.0)

    Xn = torch.tensor(Xn, dtype=torch.float32, device=device)
    M  = torch.tensor(obs, dtype=torch.float32, device=device)

    # Quadratic beta schedule (CSDI)
    beta  = torch.tensor(np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2,
                         dtype=torch.float32, device=device)
    alpha = 1.0 - beta
    abar  = torch.cumprod(alpha, dim=0)

    net = _TabCSDINet(dim, num_steps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Target ~2000 gradient steps; early stopping enforces the practical upper bound.
    steps_per_epoch = max(1, n // batch_size)
    epochs = max(20, math.ceil(2000 / steps_per_epoch))

    net.train()
    best_loss = float('inf')
    no_improve = 0
    for _ in range(epochs):
        ep_loss = 0.0
        count   = 0
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx    = perm[i:i + batch_size]
            x0, m  = Xn[idx], M[idx]
            b      = x0.size(0)

            t     = torch.randint(0, num_steps, (b,), device=device)
            noise = torch.randn_like(x0)
            a     = abar[t].unsqueeze(1)
            x_t   = a.sqrt() * x0 + (1 - a).sqrt() * noise

            # Keep a random fraction of observed entries as conditioning
            ratio        = torch.rand(b, 1, device=device)
            cond_mask    = m * (torch.rand_like(m) < ratio).float()
            target_mask  = m - cond_mask
            cond_value   = cond_mask * x0
            noisy_target = (1 - cond_mask) * x_t

            pred  = net(cond_value, noisy_target, cond_mask, t)
            denom = target_mask.sum().clamp(min=1.0)
            loss  = (((noise - pred) ** 2) * target_mask).sum() / denom

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            count   += 1

        ep_loss /= count
        if ep_loss < best_loss * (1 - 1e-3):
            best_loss  = ep_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 20:
            break

    net.eval()
    S      = n_samples
    cv     = M * Xn                                                      # (N, D)
    cv_exp = cv.unsqueeze(0).expand(S, -1, -1).reshape(S * n, dim)      # (S*N, D)
    cm_exp = M.unsqueeze(0).expand(S, -1, -1).reshape(S * n, dim)

    with torch.no_grad():
        current = torch.randn(S * n, dim, device=device)
        for t in reversed(range(num_steps)):
            noisy_target = (1 - cm_exp) * current
            t_vec = torch.full((S * n,), t, dtype=torch.long, device=device)
            pred  = net(cv_exp, noisy_target, cm_exp, t_vec)

            c1      = 1.0 / alpha[t].sqrt()
            c2      = (1 - alpha[t]) / (1 - abar[t]).sqrt()
            current = c1 * (current - c2 * pred)
            if t > 0:
                sigma   = ((1 - abar[t - 1]) / (1 - abar[t]) * beta[t]).sqrt()
                current = current + sigma * torch.randn_like(current)
            # Hold observed entries at their true values each step
            current = cm_exp * cv_exp + (1 - cm_exp) * current

    out = current.reshape(S, n, dim).mean(dim=0).cpu().numpy()

    # Denormalize and restore observed values exactly
    out = out * col_std + col_mean
    out[obs.astype(bool)] = X[obs.astype(bool)]
    return out

def _sinkhorn_cost(x, y, eps, n_iters=20):
    """Entropic OT cost between two point sets (uniform weights, squared-Euclidean cost)."""
    C = torch.cdist(x, y) ** 2
    n, m = C.shape
    log_a = torch.full((n,), -math.log(n), device=x.device)
    log_b = torch.full((m,), -math.log(m), device=x.device)
    # Dual potentials are computed without tracking gradients — the gradient of the
    # OT cost w.r.t. x flows through C alone (Danskin's theorem at optimality).
    with torch.no_grad():
        f = torch.zeros(n, device=x.device)
        g = torch.zeros(m, device=x.device)
        for _ in range(n_iters):
            f = -eps * torch.logsumexp((g.unsqueeze(0) - C) / eps + log_b.unsqueeze(0), dim=1)
            g = -eps * torch.logsumexp((f.unsqueeze(1) - C) / eps + log_a.unsqueeze(1), dim=0)
        # Compute P inside no_grad so it is detached; gradient then flows through C alone
        # (Danskin's theorem: ∂W/∂x = P · ∂C/∂x, not P·(1−C/ε)·∂C/∂x)
        log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps + log_a.unsqueeze(1) + log_b.unsqueeze(0)
        P = torch.exp(log_P)
    return (P * C).sum()


def _sinkhorn_divergence(x, y, eps, n_iters=20):
    return (_sinkhorn_cost(x, y, eps, n_iters)
            - 0.5 * _sinkhorn_cost(x, x, eps, n_iters)
            - 0.5 * _sinkhorn_cost(y, y, eps, n_iters))


def impute_ot(data, seed, n_iter=OT_ITERATIONS, batch_size=128,
              n_pairs=1, lr=1e-2, sinkhorn_iters=20):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(data, dtype=np.float64).copy()
    n, dim = X.shape
    obs = (~np.isnan(X)).astype(np.float64)

    # Standardize per column (OT is scale-sensitive)
    col_mean = np.nanmean(X, axis=0)
    col_std  = np.nanstd(X, axis=0)
    col_std[col_std == 0] = 1.0
    Xn = (X - col_mean) / col_std

    M     = torch.tensor(obs, dtype=torch.float32, device=device)
    X_obs = torch.tensor(np.nan_to_num(Xn, nan=0.0), dtype=torch.float32, device=device)

    # Learnable parameter for the missing entries only; observed stay exact
    init    = np.nan_to_num(Xn, nan=0.0) + 0.1 * np.random.randn(n, dim) * (1 - obs)
    X_param = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.RMSprop([X_param], lr=lr)
    bs = max(1, min(batch_size, n // 2))

    # Scale iterations so each data point is seen ~50 times; cap at the configured max.
    n_iter = max(200, min(n_iter, (n // bs) * 50))

    # Adaptive entropic scale from a sample of pairwise costs.
    # Use only fully observed columns so missing-value zeros don't bias distances down.
    with torch.no_grad():
        obs_cols = torch.tensor(obs.mean(axis=0) == 1.0, device=device)
        s_full   = X_obs[:, obs_cols] if obs_cols.any() else X_obs
        s        = s_full[torch.randperm(n, device=device)[:min(n, 256)]]
        eps      = float(0.05 * torch.cdist(s, s).pow(2).median().clamp(min=1e-3))

    prev_params = None
    for it in range(n_iter):
        X_imp = X_obs * M + X_param * (1 - M)   # observed exact, missing learnable
        loss = 0.0
        for _ in range(n_pairs):
            perm   = torch.randperm(n, device=device)
            i1, i2 = perm[:bs], perm[bs:2 * bs]
            loss = loss + _sinkhorn_divergence(X_imp[i1], X_imp[i2], eps, sinkhorn_iters)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Early stopping: check convergence of imputed values every 25 steps.
        if (it + 1) % 25 == 0:
            with torch.no_grad():
                curr = X_param.data.clone()
            if prev_params is not None:
                rel_change = torch.norm(curr - prev_params) / (torch.norm(prev_params) + 1e-8)
                if rel_change < 1e-3:
                    break
            prev_params = curr

    with torch.no_grad():
        out = (X_obs * M + X_param * (1 - M)).cpu().numpy()

    # Denormalize and restore observed values exactly
    out = out * col_std + col_mean
    out[obs.astype(bool)] = X[obs.astype(bool)]
    return out

IMPUTATION = {
    CONSTANT: {FUNCTION: lambda data, _: impute_simple(data, 'constant', 0), DETERMINISTIC: True},
    MEAN: {FUNCTION: lambda data, _: impute_simple(data, 'mean'), DETERMINISTIC: True},
    MEDIAN: {FUNCTION: lambda data, _: impute_simple(data, 'median'), DETERMINISTIC: True},
    KNN: {FUNCTION: lambda data, _: impute_knn(data), DETERMINISTIC: True},
    RF: {FUNCTION: lambda data, seed: impute_random_forest(data, seed), DETERMINISTIC: False},
    MICE: {FUNCTION: lambda data, seed: impute_mice(data, seed), DETERMINISTIC: False},
    GAIN: {FUNCTION: lambda data, seed: impute_gain(data, seed), DETERMINISTIC: False},
    TABCSDI: {FUNCTION: lambda data, seed: impute_tabcsdi(data, seed), DETERMINISTIC: False},
    OTIMPUTE: {FUNCTION: lambda data, seed: impute_ot(data, seed), DETERMINISTIC: False},
}