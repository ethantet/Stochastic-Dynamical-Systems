import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Dynamical systems ──────────────────────────────────────────────────

def lorenz(s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])

def chen(s, a=35.0, b=3.0, c=28.0):
    x, y, z = s
    return np.array([a*(y-x), (c-a)*x - x*z + c*y, x*y - b*z])

def lu(s, a=36.0, b=3.0, c=20.0):
    x, y, z = s
    return np.array([a*(y-x), c*y - x*z, x*y - b*z])

def rk4(state, dt, f):
    k1 = f(state)
    k2 = f(state + dt/2*k1)
    k3 = f(state + dt/2*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def logistic(x): return 4.0*x*(1 - x)
def tent(x):     return 2.0*np.minimum(x, 1 - x)

def reflect(x):
    # fold back into [0,1] by reflection rather than clipping
    m = np.mod(x, 2.0)
    return np.where(m > 1, 2.0 - m, m)


# ── Simulation ─────────────────────────────────────────────────────────

def run_ode(f, n_steps=100000, dt=0.01, sigma=0.0,
            burnin=10000, seed=42, ic=(1.0, 1.0, 1.0)):
    np.random.seed(seed)
    s = np.array(ic, dtype=float)
    out = []
    for t in range(n_steps + burnin):
        s = rk4(s, dt, f)
        if sigma > 0:
            s += np.random.normal(0, sigma, 3)
        if t >= burnin:
            out.append(s.copy())
    return np.array(out)

def run_map(f, n_steps=100000, sigma=0.0, burnin=5000, seed=42, x0=0.4):
    np.random.seed(seed)
    x = x0
    out = np.empty(n_steps)
    for t in range(n_steps + burnin):
        x = f(x)
        if sigma > 0:
            x = reflect(x + np.random.normal(0, sigma))
        if t >= burnin:
            out[t - burnin] = x
    return out

def run_ode_corr(f, noise_seq, dt=0.01, burnin=10000, ic=(1.0, 1.0, 1.0)):
    s = np.array(ic, dtype=float)
    for _ in range(burnin):
        s = rk4(s, dt, f)
    out = np.empty((len(noise_seq), 3))
    for t, n in enumerate(noise_seq):
        s = rk4(s, dt, f)
        s += n
        out[t] = s
    return out

def run_map_corr(f, noise_seq, burnin=5000, x0=0.4):
    x = x0
    for _ in range(burnin):
        x = f(x)
    out = np.empty(len(noise_seq))
    for t, n in enumerate(noise_seq):
        x = f(x)
        x = reflect(x + n[0])
        out[t] = x
    return out


# ── Correlated noise generators ────────────────────────────────────────

def ar1_noise(n, sigma, phi, dim=1, seed=42):
    # AR(1): eps_t = phi*eps_{t-1} + sqrt(1-phi^2)*N(0,sigma^2)
    # sqrt(1-phi^2) keeps marginal variance = sigma^2 for all phi
    rng = np.random.default_rng(seed)
    noise = np.zeros((n, dim))
    eps = np.zeros(dim)
    sc = sigma * np.sqrt(max(1 - phi**2, 0))
    for t in range(n):
        eps = phi*eps + sc*rng.standard_normal(dim)
        noise[t] = eps
    return noise

def fbm_noise(n, sigma, H, dim=1, seed=42):
    # fBm increments via Davies-Harte circulant embedding, O(n log n)
    # H=0.5 -> i.i.d., H>0.5 -> persistent, H<0.5 -> anti-persistent
    rng = np.random.default_rng(seed)

    def cov(k, H):
        k = np.asarray(k, dtype=float)
        return 0.5*(np.abs(k+1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k-1)**(2*H))

    m = 2*n
    row = np.zeros(m)
    row[:n] = cov(np.arange(n), H)
    row[n+1:] = row[n-1:0:-1]
    lam = np.maximum(np.real(np.fft.fft(row)), 0)

    noise = np.zeros((n, dim))
    for d in range(dim):
        z = rng.standard_normal(m) + 1j*rng.standard_normal(m)
        inc = np.real(np.fft.fft(np.sqrt(lam)*z))[:n]
        s = np.std(inc)
        if s > 0: inc *= sigma/s
        noise[:, d] = inc
    return noise


# ── MMD² distance ──────────────────────────────────────────────────────

N_SUB = 2000

def mmd2(X, Y, bw):
    def K(A, B):
        return np.exp(-((A[:, None] - B[None])**2).sum(-1) / (2*bw**2))
    return float(K(X,X).mean() + K(Y,Y).mean() - 2*K(X,Y).mean())

def bandwidth(samples, seed=0):
    rng = np.random.default_rng(seed)
    sub = samples[rng.choice(len(samples), size=min(500, len(samples)), replace=False)]
    if sub.ndim == 1: sub = sub[:, None]
    d = np.sqrt(((sub[:, None] - sub[None])**2).sum(-1))
    pos = d[d > 0]
    return float(np.median(pos)) if len(pos) > 0 else 1.0

def sub(arr, seed=0):
    rng = np.random.default_rng(seed)
    if arr.ndim == 1: arr = arr[:, None]
    return arr[rng.choice(len(arr), size=min(N_SUB, len(arr)), replace=False)]

def arcsine_samples(n, seed=0):
    u = np.random.default_rng(seed).uniform(0, 1, n)
    return np.sin(np.pi*u/2)**2

def uniform_samples(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, n)

def arcsine_density(x, eps=1e-9):
    return 1.0/(np.pi*np.sqrt(np.clip(x, eps, 1-eps)*(1 - np.clip(x, eps, 1-eps))))

def uniform_density(x):
    return np.ones_like(x)


# ── System registry ────────────────────────────────────────────────────

SYSTEMS = {
    "lorenz":   dict(type="ode", f=lorenz, dt=0.01,
                     ranges=((-30,30),(-30,30),(0,60)),
                     sigma_A=0.1, color="steelblue", label="Lorenz"),
    "chen":     dict(type="ode", f=chen, dt=0.005,
                     ranges=((-35,35),(-35,35),(0,65)),
                     sigma_A=0.1, color="darkorange", label="Chen"),
    "lu":       dict(type="ode", f=lu, dt=0.005,
                     ranges=((-30,30),(-30,30),(0,45)),
                     sigma_A=0.1, color="seagreen", label="Lu"),
    "logistic": dict(type="map", f=logistic,
                     ref_fn=arcsine_samples, density=arcsine_density,
                     sigma_A=0.05, color="steelblue",
                     label="Logistic (r=4)"),
    "tent":     dict(type="map", f=tent,
                     ref_fn=uniform_samples, density=uniform_density,
                     sigma_A=0.05, color="darkorange",
                     label="Tent (μ=2)"),
}


# ── Console prompt ─────────────────────────────────────────────────────

print("Natural Measure Experiments")
print(f"Systems: {', '.join(SYSTEMS.keys())}")
while True:
    SYSTEM = input("System: ").strip().lower()
    if SYSTEM in SYSTEMS: break
    print(f"  Choose from: {', '.join(SYSTEMS.keys())}")

print("Experiments:  1 = i.i.d. noise convergence (all systems)")
print("              2 = correlated noise — AR(1) + fBm (all systems)")
while True:
    try:
        EXP = int(input("Experiment (1 or 2): ").strip())
        if EXP in [1, 2]: break
        print("  Enter 1 or 2.")
    except ValueError:
        print("  Enter a number.")

cfg = SYSTEMS[SYSTEM]
print(f"\nSystem: {cfg['label']}   Experiment: {EXP}\n")


# ── Ground truth μ* ────────────────────────────────────────────────────

if cfg["type"] == "ode":
    print("Computing μ* (300k steps, two initial conditions)...")
    mu_star  = run_ode(cfg["f"], n_steps=300000, dt=cfg["dt"], burnin=10000)
    mu_star2 = run_ode(cfg["f"], n_steps=300000, dt=cfg["dt"], burnin=10000,
                       ic=(5.0, -3.0, 20.0))
    BW = bandwidth(mu_star)
    NOISE_FLOOR = mmd2(sub(mu_star), sub(mu_star2), BW)
    print(f"  Bandwidth: {BW:.4f}  |  Noise floor: {NOISE_FLOOR:.6f}\n")

    def dist(samples):
        return mmd2(sub(samples), sub(mu_star, seed=3), BW)

else:
    print("Ground truth: exact closed-form natural measure\n")
    _ref = cfg["ref_fn"](300000, seed=0)[:, None]
    _ref2 = cfg["ref_fn"](300000, seed=1)[:, None]
    BW = bandwidth(_ref)
    NOISE_FLOOR = mmd2(sub(_ref), sub(_ref2, seed=1), BW)
    print(f"  Bandwidth: {BW:.4f}  |  Noise floor: {NOISE_FLOOR:.6f}\n")

    def dist(samples):
        if samples.ndim == 1: samples = samples[:, None]
        return mmd2(sub(samples), sub(_ref, seed=3), BW)


# ── Experiment 1: i.i.d. noise convergence ────────────────────────────

if EXP == 1:
    if cfg["type"] == "ode":
        sigmas = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    else:
        sigmas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    N_SEEDS = 3

    print("Running noisy simulations...")
    dists = {}
    samples_cache = {}
    for s in sigmas:
        if cfg["type"] == "ode":
            runs = [run_ode(cfg["f"], n_steps=100000, dt=cfg["dt"], sigma=s, seed=k)
                    for k in range(N_SEEDS)]
            samp = np.vstack(runs)
        else:
            runs = [run_map(cfg["f"], n_steps=100000, sigma=s, seed=k)
                    for k in range(N_SEEDS)]
            samp = np.concatenate(runs)
        samples_cache[s] = samp
        dists[s] = dist(samp)
        print(f"  σ = {s:<8.4f} | MMD² = {dists[s]:.6f}")

    # plot: distributions at selected sigma values
    show = [sigmas[0], sigmas[len(sigmas)//3], sigmas[2*len(sigmas)//3], sigmas[-1]]
    show = sorted(set(show), reverse=True)[:4]
    colors = ['#e74c3c','#e67e22','#2ecc71','#3498db']

    if cfg["type"] == "ode":
        xr, yr, zr = cfg["ranges"]
        fig, axes = plt.subplots(1, 5, figsize=(24, 5))
        fig.suptitle(f'{cfg["label"]}: Stationary Distributions Converging to μ* as σ → 0',
                     fontweight='bold')
        for ax, s, c in zip(axes[:4], show, colors):
            pts = samples_cache[s]
            ax.scatter(pts[::5, 0], pts[::5, 2], s=0.1, alpha=0.2, color=c)
            ax.set_title(f'σ = {s}\nMMD² = {dists[s]:.4f}')
            ax.set_xlabel('x'); ax.set_ylabel('z')
            ax.set_xlim(*xr); ax.set_ylim(*zr)
        axes[4].scatter(mu_star[::5,0], mu_star[::5,2], s=0.05, alpha=0.15, color='#2c3e50')
        axes[4].set_title('σ = 0 (true μ*)', fontweight='bold')
        axes[4].set_xlabel('x'); axes[4].set_ylabel('z')
        axes[4].set_xlim(*xr); axes[4].set_ylim(*zr)
    else:
        xg = np.linspace(0.001, 0.999, 500)
        fig, axes = plt.subplots(1, 5, figsize=(24, 4))
        fig.suptitle(f'{cfg["label"]}: Stationary Distributions vs Exact Natural Measure',
                     fontweight='bold')
        for ax, s, c in zip(axes[:4], show, colors):
            ax.hist(samples_cache[s], bins=80, range=(0,1), density=True,
                    color=c, alpha=0.6, edgecolor='none')
            ax.plot(xg, cfg["density"](xg), 'k-', lw=2, label='exact')
            ax.set_title(f'σ = {s}\nMMD² = {dists[s]:.4f}')
            ax.set_xlabel('x'); ax.legend(fontsize=8)
        smallest = min(sigmas)
        axes[4].hist(samples_cache[smallest], bins=80, range=(0,1), density=True,
                     color='#2c3e50', alpha=0.6, edgecolor='none')
        axes[4].plot(xg, cfg["density"](xg), 'k-', lw=2, label='exact')
        axes[4].set_title(f'σ = {smallest} (smallest)', fontweight='bold')
        axes[4].set_xlabel('x'); axes[4].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{SYSTEM}_exp1_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # convergence curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(dists.keys()), list(dists.values()), 'o-',
            color=cfg["color"], lw=2.5, markersize=9,
            markerfacecolor='white', markeredgewidth=2.5)
    for s, d in dists.items():
        ax.annotate(f'{d:.4f}', (s, d), textcoords='offset points',
                    xytext=(0, 13), ha='center', fontsize=8)
    ax.axhspan(0, NOISE_FLOOR, color='gray', alpha=0.15)
    ax.axhline(NOISE_FLOOR, color='gray', ls=':', lw=2,
               label=f'Noise floor ({NOISE_FLOOR:.4f})')
    ax.set_xscale('log')
    ax.set_xlabel('σ (log scale)'); ax.set_ylabel('MMD² to μ*')
    ax.set_title(f'{cfg["label"]}: Stationary Distributions Converge to μ* as σ → 0',
                 fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{SYSTEM}_exp1_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'σ':<12} {'MMD²':<18} {'Interpretation'}")
    print("-"*55)
    for s, d in dists.items():
        if d <= NOISE_FLOOR*1.5:  note = "at noise floor"
        elif d > 0.5:             note = "far"
        elif d > 0.1:             note = "converging"
        else:                     note = "close to floor"
        print(f"{s:<12} {d:<18.6f} {note}")


# ── Experiment 2: correlated noise ────────────────────────────────────

elif EXP == 2:
    N_STEPS_C = 100000
    N_SEEDS_C = 3

    def run_corr(noise_type, param, sigma, seed):
        dim = 3 if cfg["type"] == "ode" else 1
        noise = ar1_noise(N_STEPS_C, sigma, phi=param, dim=dim, seed=seed) \
                if noise_type == "ar1" \
                else fbm_noise(N_STEPS_C, sigma, H=param, dim=dim, seed=seed)
        if cfg["type"] == "ode":
            traj = run_ode_corr(cfg["f"], noise, dt=cfg["dt"])
            return dist(traj)
        else:
            traj = run_map_corr(cfg["f"], noise)
            return dist(traj)

    def avg(noise_type, param, sigma):
        return np.mean([run_corr(noise_type, param, sigma, s) for s in range(N_SEEDS_C)])

    # Part A: fixed sigma, sweep correlation
    SA = cfg["sigma_A"]
    phis = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    Hs   = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    print(f"Part A: fixed σ = {SA}, sweeping correlation...")
    ar1_A = {}
    for p in phis:
        ar1_A[p] = avg("ar1", p, SA)
        print(f"  AR(1) φ={p:.2f} | MMD² = {ar1_A[p]:.6f}")

    fbm_A = {}
    for H in Hs:
        fbm_A[H] = avg("fbm", H, SA)
        print(f"  fBm  H={H:.2f}  | MMD² = {fbm_A[H]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{cfg["label"]}: Effect of Correlation Strength at Fixed σ = {SA}',
                 fontweight='bold')

    ax = axes[0]
    ax.plot(list(ar1_A.keys()), list(ar1_A.values()), 'o-',
            color=cfg["color"], lw=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2)
    for p, d in ar1_A.items():
        ax.annotate(f'{d:.4f}', (p, d), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8)
    ax.axhspan(0, NOISE_FLOOR, color='gray', alpha=0.15)
    ax.axhline(NOISE_FLOOR, color='gray', ls=':', lw=2,
               label=f'Noise floor ({NOISE_FLOOR:.4f})')
    ax.set_xlabel('AR(1) φ  (0 = i.i.d.)'); ax.set_ylabel('MMD² to μ*')
    ax.set_title('AR(1)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(list(fbm_A.keys()), list(fbm_A.values()), 's-',
            color=cfg["color"], lw=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2)
    for H, d in fbm_A.items():
        ax.annotate(f'{d:.4f}', (H, d), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8)
    ax.axhspan(0, NOISE_FLOOR, color='gray', alpha=0.15)
    ax.axhline(NOISE_FLOOR, color='gray', ls=':', lw=2,
               label=f'Noise floor ({NOISE_FLOOR:.4f})')
    ax.axvline(0.5, color='green', ls='--', alpha=0.6, label='H=0.5 (i.i.d.)')
    ax.set_xlabel('fBm H  (0.5 = i.i.d.)'); ax.set_ylabel('MMD² to μ*')
    ax.set_title('fBm'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{SYSTEM}_exp2_partA.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Part B: fixed correlation, sweep sigma
    if cfg["type"] == "ode":
        sigmas = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    else:
        sigmas = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

    ar1_levels = {"low (φ=0.3)": 0.3, "medium (φ=0.7)": 0.7, "high (φ=0.95)": 0.95}
    fbm_levels = {"anti-persist (H=0.3)": 0.3, "i.i.d. (H=0.5)": 0.5,
                  "persistent (H=0.8)": 0.8, "strong persist (H=0.95)": 0.95}

    print("\nPart B: sweeping σ at fixed correlation levels...")
    ar1_B, fbm_B = {}, {}
    for lbl, phi in ar1_levels.items():
        ar1_B[lbl] = {}
        print(f"  AR(1) {lbl}:")
        for s in sigmas:
            ar1_B[lbl][s] = avg("ar1", phi, s)
            print(f"    σ={s:.4f} | MMD²={ar1_B[lbl][s]:.6f}")
    for lbl, H in fbm_levels.items():
        fbm_B[lbl] = {}
        print(f"  fBm {lbl}:")
        for s in sigmas:
            fbm_B[lbl][s] = avg("fbm", H, s)
            print(f"    σ={s:.4f} | MMD²={fbm_B[lbl][s]:.6f}")

    colors_B = ['#e74c3c','#e67e22','#2ecc71','#3498db']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{cfg["label"]}: Does Noise-Limit Convergence Hold Under Correlated Noise?',
                 fontweight='bold')

    ax = axes[0]
    for (lbl, _), c in zip(ar1_levels.items(), colors_B):
        ss = list(ar1_B[lbl].keys()); dd = list(ar1_B[lbl].values())
        ax.plot(ss, dd, 'o-', color=c, lw=2, markersize=7,
                markerfacecolor='white', markeredgewidth=2, label=lbl)
    ax.axhspan(0, NOISE_FLOOR, color='gray', alpha=0.15)
    ax.axhline(NOISE_FLOOR, color='gray', ls=':', lw=2,
               label=f'Noise floor ({NOISE_FLOOR:.4f})')
    ax.set_xscale('log'); ax.set_xlabel('σ (log scale)')
    ax.set_ylabel('MMD² to μ*'); ax.set_title('AR(1)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (lbl, _), c in zip(fbm_levels.items(), colors_B):
        ss = list(fbm_B[lbl].keys()); dd = list(fbm_B[lbl].values())
        ax.plot(ss, dd, 's-', color=c, lw=2, markersize=7,
                markerfacecolor='white', markeredgewidth=2, label=lbl)
    ax.axhspan(0, NOISE_FLOOR, color='gray', alpha=0.15)
    ax.axhline(NOISE_FLOOR, color='gray', ls=':', lw=2,
               label=f'Noise floor ({NOISE_FLOOR:.4f})')
    ax.set_xscale('log'); ax.set_xlabel('σ (log scale)')
    ax.set_ylabel('MMD² to μ*'); ax.set_title('fBm')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{SYSTEM}_exp2_partB.png'),
                dpi=150, bbox_inches='tight')
    plt.show()