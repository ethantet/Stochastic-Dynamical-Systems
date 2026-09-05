import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt


# ── Dynamical systems ──────────────────────────────────────────────────

def rk4(f, dt):
    def step(s):
        k1 = f(s)
        k2 = f(s + 0.5*dt*k1)
        k3 = f(s + 0.5*dt*k2)
        k4 = f(s + dt*k3)
        return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return step

def lorenz(s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([sigma*(y-x), x*(rho-z)-y, x*y-beta*z], -1)

def chen(s, a=35.0, b=3.0, c=28.0):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([a*(y-x), (c-a)*x - x*z + c*y, x*y - b*z], -1)

def lu(s, a=36.0, b=3.0, c=20.0):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([a*(y-x), -x*z + c*y, x*y - b*z], -1)


SYSTEMS = {
    "lorenz": dict(
        dim=3, is_map=False, dt=0.02,
        step=rk4(lorenz, 0.02),
        ic=lambda rng, n: np.stack([rng.uniform(-20,20,n), rng.uniform(-20,20,n), rng.uniform(7,47,n)], -1),
        scale=0.01, dither=0.0,
        split=lambda m: m[:, 0] >= 0,
        plot_dims=(0, 2), plot_labels=("x", "z"),
    ),
    "chen": dict(
        dim=3, is_map=False, dt=0.005,
        step=rk4(chen, 0.005),
        ic=lambda rng, n: np.stack([rng.uniform(-15,15,n), rng.uniform(-15,15,n), rng.uniform(5,45,n)], -1),
        scale=0.01, dither=0.0,
        split=lambda m: m[:, 0] >= 0,
        plot_dims=(0, 2), plot_labels=("x", "z"),
    ),
    "lu": dict(
        dim=3, is_map=False, dt=0.005,
        step=rk4(lu, 0.005),
        ic=lambda rng, n: np.stack([rng.uniform(-15,15,n), rng.uniform(-15,15,n), rng.uniform(5,40,n)], -1),
        scale=0.01, dither=0.0,
        split=lambda m: m[:, 0] >= 0,
        plot_dims=(0, 2), plot_labels=("x", "z"),
    ),
    "logistic_map": dict(
        dim=1, is_map=True, dt=1.0,
        step=lambda x: 4.0*x*(1.0-x),
        ic=lambda rng, n: rng.uniform(0.01, 0.99, n).reshape(n, 1),
        scale=1.0, dither=1e-9,
        split=lambda m: m[:, 0] >= 0.5,
        plot_dims=(0,), plot_labels=("x",),
        esn_overrides=dict(spectral_radius=1.0, input_rescale=1.5),
    ),
    "tent_map": dict(
        dim=1, is_map=True, dt=1.0,
        step=lambda x: 2.0*np.minimum(x, 1.0-x),
        ic=lambda rng, n: rng.uniform(0.01, 0.99, n).reshape(n, 1),
        scale=1.0, dither=1e-9,
        split=lambda m: m[:, 0] >= 0.5,
        plot_dims=(0,), plot_labels=("x",),
        esn_overrides=dict(spectral_radius=0.9, input_rescale=2.5),
    ),
}


def simulate(spec, ic, n_steps, rng):
    out = np.empty((n_steps,) + ic.shape, dtype=np.float64)
    out[0] = ic
    s = ic.copy()
    for t in range(1, n_steps):
        s = spec["step"](s)
        if spec["is_map"] and spec["dither"] > 0:
            s = np.clip(s + rng.normal(0.0, spec["dither"], s.shape), 0.0, 1.0)
        out[t] = s
    return out


# ── Echo State Network ─────────────────────────────────────────────────

class ESN:
    def __init__(self, dim, size, rng, spectral_radius=1.2,
                 input_rescale=13/3, ridge=10**(-14.5)):
        self.size = size
        self.ridge = ridge
        A = rng.uniform(-0.5, 0.5, size=(size, size))
        A *= spectral_radius / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.C = rng.uniform(-0.5, 0.5, size=(size, dim)) * input_rescale
        self.W = None

    def _update(self, x, u):
        return np.tanh(x @ self.A.T + u @ self.C.T)

    def fit(self, U):
        T, n, dim = U.shape
        washout = T // 3
        XtX = np.zeros((self.size, self.size))
        XtY = np.zeros((self.size, dim))
        x = np.zeros((n, self.size))
        for t in range(T - 1):
            x = self._update(x, U[t])
            if t + 1 > washout:
                XtX += x.T @ x
                XtY += x.T @ U[t + 1]
        self.W = np.linalg.solve(XtX + self.ridge*np.eye(self.size), XtY).T
        return float(np.sqrt(np.mean((x @ self.W.T - U[T-1])**2)))

    def warmup(self, U):
        x = np.zeros((U.shape[1], self.size))
        for t in range(U.shape[0]):
            x = self._update(x, U[t])
        return x

    def run(self, x0, n_steps, clip=None, scale=1.0):
        out = np.empty((n_steps, x0.shape[0], self.W.shape[0]))
        u = x0 @ self.W.T
        if clip is not None:
            u = np.clip(u, clip[0]*scale, clip[1]*scale)
        out[0] = u
        x = x0
        for i in range(1, n_steps):
            x = self._update(x, u)
            u = x @ self.W.T
            if clip is not None:
                u = np.clip(u, clip[0]*scale, clip[1]*scale)
            out[i] = u
        return out


# ── MMD and hypothesis tests ───────────────────────────────────────────

def mmd2(X, Y, bw):
    def K(A, B):
        d2 = ((A[:, None] - B[None])**2).sum(-1)
        return np.exp(-d2 / (2*bw**2))
    return K(X,X).mean() + K(Y,Y).mean() - 2*K(X,Y).mean()

def thresholds(n, alpha=0.05, kappa=1.0, eps2=0.1):
    # Critical values for the two-sample MMD test (Louw & Ortega 2025)
    ca = (np.sqrt(2*kappa/n) * (1 + np.sqrt(2*np.log(1/alpha))))**2
    b  = np.sqrt(eps2) - 2*np.sqrt(kappa/n)*(2 + np.sqrt(np.log(2/alpha)))
    cb = max(b, 0.0)**2
    return ca, cb


# ── Main experiment loop ───────────────────────────────────────────────

def run(name, args, out_dir, log):
    spec = SYSTEMS[name]
    rng  = np.random.default_rng(args.seed)
    t0   = time.time()
    sc   = spec["scale"]
    print(f"\n=== {name} ===")

    # train
    net = ESN(spec["dim"], args.neurons, rng, **spec.get("esn_overrides", {}))
    ic  = spec["ic"](rng, args.train_traj)
    rmse = net.fit(simulate(spec, ic, args.train_steps, rng) * sc)
    print(f"  trained: RMSE (scaled) = {rmse:.2e}  [{time.time()-t0:.0f}s]")

    # autonomous prediction
    ic      = spec["ic"](rng, args.pred_traj)
    truth   = simulate(spec, ic, args.total_steps, rng)
    x0      = net.warmup(truth[:args.warmup_steps] * sc)
    clip    = (0.0, 1.0) if spec["is_map"] else None
    pred    = net.run(x0, args.total_steps - args.warmup_steps, clip=clip, scale=sc) / sc
    print(f"  prediction done  [{time.time()-t0:.0f}s]")

    # lobe split
    mask      = spec["split"](truth[args.warmup_steps])
    i1_pool   = np.where(mask)[0]
    i2_pool   = np.where(~mask)[0]
    n_lobe    = min(args.lobe_size, len(i1_pool), len(i2_pool))
    i1        = rng.permutation(i1_pool)[:n_lobe]
    i2        = rng.permutation(i2_pool)[:n_lobe]
    mu1       = truth[args.warmup_steps:, i1]
    mu2       = truth[args.warmup_steps:, i2]
    mu1_hat   = pred[:, i1]

    # kernel bandwidth (median heuristic)
    pool = truth[args.warmup_steps:, rng.choice(np.r_[i1, i2], size=min(100, 2*n_lobe), replace=False)]
    sub  = pool.reshape(-1, spec["dim"])
    sub  = sub[rng.choice(len(sub), size=min(2000, len(sub)), replace=False)]
    dists = np.sqrt(((sub[:, None] - sub[None])**2).sum(-1))
    bw    = float(np.median(dists[dists > 0]))

    ca, cb = thresholds(n_lobe)

    # MMD curves
    n_pred = mu1.shape[0]
    idx    = np.arange(0, n_pred, max(1, n_pred // 150))
    mmd_true = np.array([mmd2(mu1[t], mu2[t],     bw) for t in idx])
    mmd_esn  = np.array([mmd2(mu1[t], mu1_hat[t], bw) for t in idx])
    pt_err   = np.linalg.norm(mu1 - mu1_hat, axis=-1).mean(axis=1)
    pt_ref   = np.linalg.norm(mu1 - mu2,     axis=-1).mean(axis=1)
    print(f"  MMD curves done  [{time.time()-t0:.0f}s]")

    # validation
    dt   = spec["dt"]
    tau  = idx * dt + args.warmup_steps * dt
    tau0 = args.warmup_steps * dt
    tau1 = args.total_steps  * dt
    stable    = tau > tau0 + 0.5*n_pred*dt
    v1        = float(mmd_true[0])
    v2a       = float(np.median(mmd_true[stable]))
    v2b       = float(np.median(mmd_esn[stable]))
    v2c       = v1/v2b if v2b > 0 else float("inf")
    tail      = max(1, len(pt_err)//6)
    v3        = float(pt_err[-tail:].mean() / pt_ref[-tail:].mean())
    ok        = lambda x: "PASS" if x else "FAIL"

    lines = [
        f"--- {name} (L={args.neurons}) ---",
        f"(V1)  initial  MMD²(μ1,μ2):     {v1:.5f}  > {ca:.5f}  {ok(v1 > ca)}",
        f"(V2a) stable   MMD²(μ1,μ2):     {v2a:.5f}  < {cb:.5f}  {ok(v2a < cb)}",
        f"(V2b) stable   MMD²(μ1,μ1_hat): {v2b:.5f}  < {cb:.5f}  {ok(v2b < cb)}",
        f"(V2c) ratio V1/V2b:              {v2c:.0f}x  ~100x  {ok(v2b > 0 and v2c > 50)}",
        f"(V3)  pt-error saturation:       {v3:.3f}   0.7–1.3  {ok(0.7 < v3 < 1.3)}",
    ]
    for l in lines: print("  " + l)
    log.extend(lines + [""])

    # Figure 1 — distribution snapshots
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (mu2[0], r"$\mu^2_\tau$"), (mu1[0], r"$\mu^1_\tau$"), (mu1_hat[0], r"$\hat\mu^1_\tau$"),
        (mu2[-1],r"$\mu^2_\tau$"), (mu1[-1],r"$\mu^1_\tau$"), (mu1_hat[-1],r"$\hat\mu^1_\tau$"),
    ]
    row_labels = [f"$\\tau$={tau0:g} (prediction start)", f"$\\tau$={tau1:g} (terminal)"]
    for i, (ax, (pts, lbl)) in enumerate(zip(axes.flat, panels)):
        if spec["dim"] >= 2:
            d0, d1 = spec["plot_dims"]
            ax.hist2d(pts[:, d0], pts[:, d1], bins=60, cmap="viridis")
            ax.set_xlabel(spec["plot_labels"][0]); ax.set_ylabel(spec["plot_labels"][1])
        else:
            ax.hist(pts[:, 0], bins=40, range=(0,1), color="tab:blue")
            ax.set_xlabel(spec["plot_labels"][0])
        ax.set_title(f"{lbl} at {row_labels[0 if i<3 else 1]}", fontsize=9)
    fig.suptitle(f"{name}: distribution snapshots (cf. paper Figure 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_fig1.png"), dpi=140)
    plt.show(); plt.close(fig)

    # Figure 2 — point errors + MMD²
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    tf     = np.arange(n_pred) * dt + tau0
    xlabel = r"time $\tau$" if not spec["is_map"] else "time step"

    axes[0].semilogy(tf, pt_ref, "tab:blue",   lw=0.7, label=r"$\mu^1_t$ vs $\mu^2_t$")
    axes[0].semilogy(tf, pt_err, "tab:orange", lw=0.7, label=r"$\mu^1_t$ vs $\hat\mu^1_t$")
    axes[0].axvline(tau0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Distance")
    axes[0].set_title("(a) Point prediction error"); axes[0].legend(fontsize=8)

    axes[1].semilogy(tau, mmd_true, "tab:blue",   lw=0.9, label=r"MMD²($\mu^1_t$, $\mu^2_t$)")
    axes[1].semilogy(tau, mmd_esn,  "tab:orange", lw=0.9, label=r"MMD²($\mu^1_t$, $\hat\mu^1_t$)")
    axes[1].axhline(ca, color="tab:blue", ls=":", lw=1, label=f"Test A ({ca:.4f})")
    axes[1].axhline(cb, color="tab:red",  ls=":", lw=1, label=f"Test B ({cb:.4f})")
    axes[1].axvline(tau0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(r"MMD²")
    axes[1].set_title("(b) MMD² vs time"); axes[1].legend(fontsize=7)

    fig.suptitle(f"{name}: point errors + MMD² over prediction horizon (cf. paper Figure 2)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_fig2.png"), dpi=140)
    plt.show(); plt.close(fig)

    print(f"  saved  [{time.time()-t0:.0f}s total]")


def main():
    ap = argparse.ArgumentParser(description="ESN climate learning — Louw & Ortega (2025)")
    ap.add_argument("--systems",     nargs="+", default=list(SYSTEMS.keys()), choices=list(SYSTEMS.keys()))
    ap.add_argument("--neurons",     type=int, default=1000)
    ap.add_argument("--train-traj",  type=int, default=100)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--pred-traj",   type=int, default=2200)
    ap.add_argument("--total-steps", type=int, default=4000)
    ap.add_argument("--warmup-steps",type=int, default=1000)
    ap.add_argument("--lobe-size",   type=int, default=1000)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--quick",       action="store_true")
    ap.add_argument("--out",         default="figures")
    args = ap.parse_args()

    if args.quick:
        args.neurons = 200; args.train_traj = 20; args.train_steps = 1200
        args.pred_traj = 500; args.total_steps = 1600
        args.warmup_steps = 400; args.lobe_size = 200

    os.makedirs(args.out, exist_ok=True)
    log = []
    for name in args.systems:
        run(name, args, args.out, log)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(log))
    print(f"\nDone. Results in ./{args.out}/")

if __name__ == "__main__":
    main()