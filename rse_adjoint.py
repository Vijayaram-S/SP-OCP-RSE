"""
Discrete adjoint (GRAPE-type backpropagation) gradients for the CFM4-discretized
noise-metric-aware control problem, plus the numerical study reported in the
revised manuscript (Section 6, "Adjoint gradients: verification, cost and use
in optimization").

The adjoint follows Theorem 5.1 of the manuscript:
    P_N = -(1/d) V tr(V^dag U_N)/|tr(V^dag U_N)|,
    P_n = Q_n^dag P_{n+1},       Q_n = exp(Omega_2,n) exp(Omega_1,n),
    dJ/du_j^n = Re tr(P_{n+1}^dag  (dQ_n/du_j^n) U_n) + lambda h l_j u_j^n / Delta_n,
with the Frechet derivative of the matrix exponential evaluated exactly by the
block-triangular identity (Van Loan 1978)
    expm([[X, E],[0, X]]) = [[e^X, Dexp_X[E]],[0, e^X]].
The block identity is dimension-independent, so the same code applies to SU(d), where d is the Hilbert-space dimension.

Usage:  python rse_adjoint.py check | timing | reopt | all
Outputs go to ./out_adjoint/
"""
import sys, os, json, time
import numpy as np
from scipy.linalg import expm

# ---- import the (unchanged) forward model, noise model and FD optimizer ----
_argv = sys.argv; sys.argv = [sys.argv[0]]          # keep the stage dispatcher idle
import rse_experiments_corrected as base
sys.argv = _argv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out_adjoint')
os.makedirs(OUT, exist_ok=True)

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
a1, a2, omega0, V = base.a1, base.a2, base.omega0, base.V


def node_hamiltonians(ux, uy, N, h, nz=None, ndx=None, ndy=None):
    """Hamiltonians at the two Gauss nodes of every step for one noise sample.
    nz/ndx/ndy: tuples (node1, node2) of length-N arrays, or None.
    Returns H1, H2 (N,2,2) and the control-coupling factors fx_r, fy_r (N,)."""
    z = lambda p, r: (p[r] if p is not None else np.zeros(N))
    fx = [1.0 + z(ndx, r) for r in (0, 1)]
    fy = [1.0 + z(ndy, r) for r in (0, 1)]
    Hs = []
    for r in (0, 1):
        Hr = ((0.5 * omega0 + z(nz, r))[:, None, None] * sz
              + (ux * fx[r])[:, None, None] * sx + (uy * fy[r])[:, None, None] * sy)
        Hs.append(Hr)
    return Hs[0], Hs[1], fx, fy


def gate_cost_and_grad(ux, uy, h, nz=None, ndx=None, ndy=None):
    """Infidelity 1-|tr(V^dag U_N)|/2 of one sample and its adjoint gradient
    with respect to (ux, uy). Returns (J, gx, gy)."""
    N = len(ux)
    H1, H2, fx, fy = node_hamiltonians(ux, uy, N, h, nz, ndx, ndy)
    Om1 = -1j * h * (a1 * H1 + a2 * H2)
    Om2 = -1j * h * (a2 * H1 + a1 * H2)
    # derivatives of Omega_1, Omega_2 with respect to u_x^n and u_y^n
    dO1 = {'x': (-1j * h * (a1 * fx[0] + a2 * fx[1]))[:, None, None] * sx,
           'y': (-1j * h * (a1 * fy[0] + a2 * fy[1]))[:, None, None] * sy}
    dO2 = {'x': (-1j * h * (a2 * fx[0] + a1 * fx[1]))[:, None, None] * sx,
           'y': (-1j * h * (a2 * fy[0] + a1 * fy[1]))[:, None, None] * sy}

    def exp_and_frechet(X, E):          # batched over steps
        Z = np.zeros_like(X)
        B = np.block([[X, E], [Z, X]])   # (N,4,4)
        M = expm(B)
        return M[:, :2, :2], M[:, :2, 2:]

    E1, D1x = exp_and_frechet(Om1, dO1['x'])
    _, D1y = exp_and_frechet(Om1, dO1['y'])
    E2, D2x = exp_and_frechet(Om2, dO2['x'])
    _, D2y = exp_and_frechet(Om2, dO2['y'])
    Q = E2 @ E1
    dQx = D2x @ E1 + E2 @ D1x
    dQy = D2y @ E1 + E2 @ D1y

    # forward sweep
    U = np.empty((N + 1, 2, 2), complex); U[0] = np.eye(2)
    for n in range(N):
        U[n + 1] = Q[n] @ U[n]
    z = np.trace(V.conj().T @ U[N])
    J = 1.0 - abs(z) / 2.0
    # backward (adjoint) sweep
    P = -0.5 * (z / abs(z)) * V
    gx = np.empty(N); gy = np.empty(N)
    for n in range(N - 1, -1, -1):
        Ph = P.conj().T
        gx[n] = np.real(np.trace(Ph @ dQx[n] @ U[n]))
        gy[n] = np.real(np.trace(Ph @ dQy[n] @ U[n]))
        P = Q[n].conj().T @ P
    return J, gx, gy


def full_cost_and_grad(u, lam, lx, ly, h, noise=None):
    """J = mean_s infid_s + lam*h*sum_n Delta_n and its adjoint gradient.
    noise: None (deterministic surrogate) or (nz, ndx, ndy) with (S,N) node arrays."""
    N = len(u) // 2
    ux, uy = u[:N], u[N:]
    if noise is None:
        samples = [(None, None, None)]
    else:
        nz, ndx, ndy = noise
        S = max(p[0].shape[0] for p in noise if p is not None)
        pick = lambda p, s: None if p is None else (p[0][s], p[1][s])
        samples = [(pick(nz, s), pick(ndx, s), pick(ndy, s)) for s in range(S)]
    J = 0.0; gx = np.zeros(N); gy = np.zeros(N)
    for (a, b, c) in samples:
        Js, gxs, gys = gate_cost_and_grad(ux, uy, h, a, b, c)
        J += Js; gx += gxs; gy += gys
    k = len(samples); J /= k; gx /= k; gy /= k
    if lam > 0:
        D = np.sqrt(lx * ux ** 2 + ly * uy ** 2)
        Ds = np.where(D > 1e-14, D, 1.0)
        J += lam * h * np.sum(D)
        gx += lam * h * lx * ux / Ds * (D > 1e-14)
        gy += lam * h * ly * uy / Ds * (D > 1e-14)
    return J, np.concatenate([gx, gy])


def adam_adjoint(lam, lx, ly, noise=None, iters=450, lr0=0.05):
    """Identical projected-Adam schedule, initial guess and bounds as the FD
    optimizer of rse_experiments_corrected.py; only the gradient differs."""
    N, T, tg = base.N, base.T, base.tgrid
    u = np.concatenate([0.5 * np.sin(2 * np.pi * tg[:-1] / T), 0.5 * np.cos(2 * np.pi * tg[:-1] / T)])
    m = np.zeros(2 * N); v = np.zeros(2 * N); b1, b2 = 0.9, 0.999
    for it in range(1, iters + 1):
        _, g = full_cost_and_grad(u, lam, lx, ly, base.h, noise)
        lr = lr0 * (0.3 if it > iters * 0.6 else 1.0) * (0.1 if it > iters * 0.85 else 1.0)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
        u = np.clip(u - lr * mh / (np.sqrt(vh) + 1e-12), -2.0, 2.0)
    return u


def fd_J_batch_factory(lam, lx, ly, noise=None):
    nz, ndx, ndy = noise if noise is not None else (None, None, None)
    N = base.N
    def J_batch(ub):
        Ux = ub[:, :N]; Uy = ub[:, N:]
        c = base.infid(*base.prop_cfm4(Ux, Uy, nz, ndx, ndy)).mean(axis=1)
        return c + lam * base.Jnoise(Ux, Uy, lx, ly)
    return J_batch


# ------------------------------------------------------------------ stages
def stage_check():
    """Gradient verification against central finite differences (eps=1e-6)."""
    rng = np.random.default_rng(4242)
    N = base.N; h = base.h
    res = {}
    train = base.tr_noise                       # 30 control-coupled training samples
    cases = [('deterministic surrogate, lambda=5e-3, l_y=4', 0.005, 1, 4, None),
             ('noise-blind, lambda=0', 0.0, 1, 1, None),
             ('ensemble cost, 30 samples', 0.0, 1, 1, train)]
    for name, lam, lx, ly, noise in cases:
        u = np.clip(0.5 * rng.standard_normal(2 * N), -2, 2)
        J, g = full_cost_and_grad(u, lam, lx, ly, h, noise)
        Jb = fd_J_batch_factory(lam, lx, ly, noise)
        eps = 1e-6
        P = np.vstack([u[None, :] + eps * np.eye(2 * N), u[None, :] - eps * np.eye(2 * N)])
        Jv = Jb(P)
        gfd = (Jv[:2 * N] - Jv[2 * N:]) / (2 * eps)
        Jcheck = float(Jb(u[None, :])[0])
        rel = float(np.linalg.norm(g - gfd) / np.linalg.norm(gfd))
        res[name] = dict(J_adjoint_code=float(J), J_forward_code=Jcheck,
                         rel_l2_error=rel, max_abs_error=float(np.max(np.abs(g - gfd))),
                         grad_norm=float(np.linalg.norm(gfd)))
        print(name, res[name], flush=True)
        if noise is None and lam > 0:
            np.savez(os.path.join(OUT, 'gradcheck.npz'), g=g, gfd=gfd)
    json.dump(res, open(os.path.join(OUT, 'gradcheck.json'), 'w'), indent=1)


def stage_timing():
    """Wall-clock cost of one gradient: adjoint vs batched FD, as a function of N.
    Uses a self-contained N so the base module's grid is not modified."""
    Ns = [50, 100, 200, 400, 800]
    out = {'N': Ns, 'adjoint_s': [], 'fd_s': [], 'rel_err': []}
    rng = np.random.default_rng(7)
    for N in Ns:
        T = base.T; h = T / N
        tg = np.linspace(0, T, N + 1)
        u = np.clip(0.5 * rng.standard_normal(2 * N), -2, 2)
        # adjoint
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            _, g = full_cost_and_grad(u, 0.005, 1, 4, h, None)
        ta = (time.perf_counter() - t0) / reps
        # batched forward-difference FD, same vectorized propagator as the paper,
        # rebuilt for this N
        saved = (base.N, base.h)
        base.N, base.h = N, h
        try:
            Jb = fd_J_batch_factory(0.005, 1, 4, None)
            eps = 3e-7
            reps_fd = 3 if N <= 400 else 1
            t0 = time.perf_counter()
            for _ in range(reps_fd):
                P = np.vstack([u[None, :] + eps * np.eye(2 * N), u[None, :]])
                Jv = Jb(P)
                gfd = (Jv[:2 * N] - Jv[-1]) / eps
            tf = (time.perf_counter() - t0) / reps_fd
        finally:
            base.N, base.h = saved
        out['adjoint_s'].append(ta); out['fd_s'].append(tf)
        out['rel_err'].append(float(np.linalg.norm(g - gfd) / np.linalg.norm(gfd)))
        print(N, ta, tf, out['rel_err'][-1], flush=True)
    json.dump(out, open(os.path.join(OUT, 'timing.json'), 'w'), indent=1)


def stage_reopt():
    """Re-optimize the four baseline strategies with adjoint gradients and evaluate
    them on the same 1000 validation realizations used in the manuscript."""
    vn = base.val_noise()
    variants = {'A': (0.0, 1, 1, False, 450), 'B': (0.005, 1, 1, False, 450),
                'C': (0.005, 1, 4, False, 450), 'D': (0.0, 1, 1, True, 300)}
    res = {}
    for nm, (lam, lx, ly, ens, iters) in variants.items():
        t0 = time.perf_counter()
        u = adam_adjoint(lam, lx, ly, base.tr_noise if ens else None, iters=iters)
        wall = time.perf_counter() - t0
        N = base.N; ux, uy = u[:N], u[N:]
        iv = base.evaluate(ux, uy, vn)
        clean = float(base.infid(*base.prop_cfm4(ux, uy, None, None, None))[0, 0])
        res[nm] = dict(**base.stats(iv), clean=clean,
                       Jnoise_aniso=float(base.Jnoise(ux, uy, 1, 4)), wall_s=wall)
        np.save(os.path.join(OUT, f'u_adjoint_{nm}.npy'), u)
        print(nm, res[nm], flush=True)
    json.dump(res, open(os.path.join(OUT, 'reopt.json'), 'w'), indent=1)


def stage_figure():
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(OUT, 'gradcheck.npz'))
    tm = json.load(open(os.path.join(OUT, 'timing.json')))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(d['gfd'], d['g'], '.', ms=4, alpha=.6, color='tab:blue',
               label='adjoint vs. finite-difference\ngradient components\n($2N=400$, anisotropic surrogate)')
    lim = np.max(np.abs(d['gfd'])) * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], 'k--', lw=1.2, label='$y=x$')
    ax[0].set_xlabel('central finite-difference gradient')
    ax[0].set_ylabel('adjoint gradient'); ax[0].legend(loc='lower right', markerscale=3, fontsize=10); ax[0].set_title('(a) gradient verification')
    ax[1].loglog(tm['N'], tm['fd_s'], 'o-', lw=2, color='tab:red', label='batched finite differences')
    ax[1].loglog(tm['N'], tm['adjoint_s'], 's-', lw=2, color='tab:blue', label='discrete adjoint')
    Ns = np.array(tm['N'], float)
    ax[1].loglog(Ns, tm['adjoint_s'][0] * Ns / Ns[0], 'k:', lw=1, label=r'$\propto N$')
    ax[1].loglog(Ns, tm['fd_s'][0] * (Ns / Ns[0]) ** 2, 'k--', lw=1, label=r'$\propto N^2$')
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax[1].xaxis.set_major_locator(FixedLocator(tm['N']))
    ax[1].xaxis.set_major_formatter(FixedFormatter([str(n) for n in tm['N']]))
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xlabel('number of time steps $N$'); ax[1].set_ylabel('wall-clock time per gradient [s]')
    ax[1].legend(); ax[1].set_title('(b) cost per gradient evaluation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure13_adjoint.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'figure13_adjoint.pdf'), bbox_inches='tight')
    print('figure saved')


if __name__ == '__main__':
    st = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if st in ('check', 'all'): stage_check()
    if st in ('timing', 'all'): stage_timing()
    if st in ('reopt', 'all'): stage_reopt()
    if st in ('figure', 'all'): stage_figure()
