
# Toy g-2 style engine with two observables, grid-based inference,
# Wilks toys, Brazilian band, MCMC validation, and Bayes factor NP vs SM.

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

try:
    import blackjax
    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False
    print("[INFO] blackjax not installed, MCMC validation will be skipped.")


# ============================================================
# 0. Configs & validation
# ============================================================

@dataclass
class MuGridConfig:
    mu_min: float = -400.0
    mu_max: float = 400.0
    n_mu: int = 401

    def validate(self):
        assert self.mu_min < self.mu_max
        assert self.n_mu > 10


@dataclass
class KappaPriorModel:
    w: float = 0.5
    mu1: float = 0.0
    sigma1: float = 40.0
    mu2: float = 150.0
    sigma2: float = 40.0
    kappa_min: float = -100.0
    kappa_max: float = 300.0

    def validate(self):
        assert 0.0 < self.w < 1.0
        assert self.sigma1 > 0.0 and self.sigma2 > 0.0
        assert self.kappa_min < self.kappa_max


@dataclass
class KappaGridConfig:
    kappa_min: float = -100.0
    kappa_max: float = 300.0
    n_kappa: int = 401

    def validate(self):
        assert self.kappa_min < self.kappa_max
        assert self.n_kappa > 10


@dataclass
class WilksConfig:
    mu_true: float = 0.0
    kappa_true: float = 0.0
    n_toys: int = 1000
    mu_grid: MuGridConfig = field(default_factory=MuGridConfig)
    kappa_grid: KappaGridConfig = field(
        default_factory=lambda: KappaGridConfig(
            kappa_min=-100.0,
            kappa_max=300.0,
            n_kappa=401,
        )
    )

    def validate(self):
        assert self.n_toys > 100
        self.mu_grid.validate()
        self.kappa_grid.validate()


# ============================================================
# 1. Chi-square CDF & quantile (df=1)
# ============================================================

def chi2_cdf_jax(q):
    return jsp.special.erf(jnp.sqrt(q / 2.0))


def chi2_quantile_jax(alpha):
    return 2.0 * (jsp.special.erfinv(alpha))**2


# ============================================================
# 2. Likelihood model: two observables
# ============================================================

def logL_two_obs(mu, kappa,
                 delta_mu_exp, sigma_mu,
                 delta_hvp_exp, sigma_hvp):
    """
    Two-observable Gaussian likelihood:
      Obs1: delta_mu_exp  ~ N(mu + kappa, sigma_mu^2)
      Obs2: delta_hvp_exp ~ N(kappa,       sigma_hvp^2)
    """
    assert sigma_mu > 0.0 and sigma_hvp > 0.0
    resid1 = delta_mu_exp - (mu + kappa)
    resid2 = delta_hvp_exp - kappa
    return (
        -0.5 * (resid1**2) / (sigma_mu**2)
        -0.5 * (resid2**2) / (sigma_hvp**2)
    )


# ============================================================
# 3. Priors
# ============================================================

def _log_gauss(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - jnp.log(s * jnp.sqrt(2.0 * jnp.pi))


def log_prior_kappa_mixture_bounded(kappa, prior_cfg: KappaPriorModel):
    """
    Truncated Gaussian mixture prior for kappa on [kappa_min, kappa_max].
    """
    prior_cfg.validate()
    inside = (kappa >= prior_cfg.kappa_min) & (kappa <= prior_cfg.kappa_max)

    logp1 = _log_gauss(kappa, prior_cfg.mu1, prior_cfg.sigma1)
    logp2 = _log_gauss(kappa, prior_cfg.mu2, prior_cfg.sigma2)

    a = jnp.log(prior_cfg.w) + logp1
    b = jnp.log(1.0 - prior_cfg.w) + logp2
    log_mix = jsp.special.logsumexp(jnp.stack([a, b]), axis=0)

    # Truncated normalization on the same support
    k_grid = jnp.linspace(prior_cfg.kappa_min, prior_cfg.kappa_max, 4001)
    logp1_g = _log_gauss(k_grid, prior_cfg.mu1, prior_cfg.sigma1)
    logp2_g = _log_gauss(k_grid, prior_cfg.mu2, prior_cfg.sigma2)
    a_g = jnp.log(prior_cfg.w) + logp1_g
    b_g = jnp.log(1.0 - prior_cfg.w) + logp2_g
    log_mix_g = jsp.special.logsumexp(jnp.stack([a_g, b_g]), axis=0)
    dk = k_grid[1] - k_grid[0]
    logZ_trunc = jsp.special.logsumexp(log_mix_g) + jnp.log(dk)

    logp = jnp.where(inside, log_mix - logZ_trunc, -jnp.inf)
    return logp


def log_prior_kappa_uniform(kappa, prior_cfg: KappaPriorModel):
    """
    Uniform prior for kappa on [kappa_min, kappa_max].
    """
    prior_cfg.validate()
    inside = (kappa >= prior_cfg.kappa_min) & (kappa <= prior_cfg.kappa_max)
    width = jnp.maximum(prior_cfg.kappa_max - prior_cfg.kappa_min, 1e-12)
    log_width = jnp.log(width)
    return jnp.where(inside, -log_width, -jnp.inf)


def log_prior_kappa_gaussian(kappa, prior_cfg: KappaPriorModel):
    """
    Truncated single Gaussian prior for kappa on [kappa_min, kappa_max].
    """
    prior_cfg.validate()
    inside = (kappa >= prior_cfg.kappa_min) & (kappa <= prior_cfg.kappa_max)
    logp_raw = _log_gauss(kappa, prior_cfg.mu1, prior_cfg.sigma1)

    k_grid = jnp.linspace(prior_cfg.kappa_min, prior_cfg.kappa_max, 4001)
    logp_grid = _log_gauss(k_grid, prior_cfg.mu1, prior_cfg.sigma1)
    dk = k_grid[1] - k_grid[0]
    logZ = jsp.special.logsumexp(logp_grid) + jnp.log(dk)

    logp = jnp.where(inside, logp_raw - logZ, -jnp.inf)
    return logp


def log_prior_mu_wide(mu, sigma_mu_prior=1e6):
    """
    Very wide Gaussian prior for mu (effectively flat on the region of interest).
    """
    assert sigma_mu_prior > 0.0
    return -0.5 * (mu / sigma_mu_prior) ** 2


# ============================================================
# 4. Grid-based profile & Bayes (two obs) + evidence
# ============================================================

def profile_and_bayes_two_obs(
    delta_mu_exp: float,
    sigma_mu: float,
    delta_hvp_exp: float,
    sigma_hvp: float,
    mu_cfg: MuGridConfig,
    kappa_grid_cfg: KappaGridConfig,
    kappa_prior_cfg: KappaPriorModel,
    log_prior_kappa_fn=log_prior_kappa_mixture_bounded,
):
    """
    Grid-based profile likelihood and Bayesian marginal posterior for mu,
    with evidence for the model where mu is free (NP hypothesis).
    """
    mu_cfg.validate()
    kappa_grid_cfg.validate()
    kappa_prior_cfg.validate()

    mu_grid = jnp.linspace(mu_cfg.mu_min, mu_cfg.mu_max, mu_cfg.n_mu)
    kappa_grid = jnp.linspace(
        kappa_grid_cfg.kappa_min,
        kappa_grid_cfg.kappa_max,
        kappa_grid_cfg.n_kappa,
    )

    mu_mesh, kappa_mesh = jnp.meshgrid(mu_grid, kappa_grid, indexing="ij")

    logL_2d = logL_two_obs(
        mu_mesh,
        kappa_mesh,
        delta_mu_exp,
        sigma_mu,
        delta_hvp_exp,
        sigma_hvp,
    )

    # Profile likelihood in mu
    logL_prof_mu = jnp.max(logL_2d, axis=1)
    logL_max = jnp.max(logL_prof_mu)
    q_prof_mu = -2.0 * (logL_prof_mu - logL_max)

    idx_prof = jnp.argmin(q_prof_mu)
    mu_hat_prof = float(mu_grid[idx_prof])

    # Bayesian marginal posterior in mu
    logPrior_kappa_2d = log_prior_kappa_fn(kappa_mesh, kappa_prior_cfg)
    logPrior_mu_2d = log_prior_mu_wide(mu_mesh)

    logPost_2d = logL_2d + logPrior_kappa_2d + logPrior_mu_2d

    dkappa = (
        kappa_grid_cfg.kappa_max - kappa_grid_cfg.kappa_min
    ) / (kappa_grid_cfg.n_kappa - 1)

    def log_marginal_mu(logPost_row):
        return jsp.special.logsumexp(logPost_row) + jnp.log(dkappa)

    logPost_marg_mu = jax.vmap(log_marginal_mu)(logPost_2d)

    dmu = (mu_cfg.mu_max - mu_cfg.mu_min) / (mu_cfg.n_mu - 1)
    logZ = jsp.special.logsumexp(logPost_marg_mu) + jnp.log(dmu)
    logPost_marg_mu_norm = logPost_marg_mu - logZ

    idx_bayes = jnp.argmax(logPost_marg_mu_norm)
    mu_hat_bayes = float(mu_grid[idx_bayes])

    if int(idx_bayes) in (0, mu_cfg.n_mu - 1):
        print(f"WARNING: Bayes mu_hat at grid boundary = {mu_hat_bayes:.2f}")

    if not jnp.isfinite(logZ):
        print("WARNING: logZ (evidence for NP) is not finite")

    return {
        "mu_grid": mu_grid,
        "kappa_grid": kappa_grid,
        "mu_mesh": mu_mesh,
        "kappa_mesh": kappa_mesh,
        "logL_2d": logL_2d,
        "logL_prof_mu": logL_prof_mu,
        "q_prof_mu": q_prof_mu,
        "logPost_marg_mu": logPost_marg_mu_norm,
        "mu_hat_prof": float(mu_hat_prof),
        "mu_hat_bayes": float(mu_hat_bayes),
        "logZ": float(logZ),
    }


# ============================================================
# 5. Evidence for SM-only (mu = 0)
# ============================================================

def evidence_sm_two_obs(
    delta_mu_exp: float,
    sigma_mu: float,
    delta_hvp_exp: float,
    sigma_hvp: float,
    kappa_grid_cfg: KappaGridConfig,
    kappa_prior_cfg: KappaPriorModel,
    log_prior_kappa_fn=log_prior_kappa_mixture_bounded,
):
    """
    Evidence for SM-only hypothesis: mu fixed to 0, integrate over kappa.
    Z_SM = ∫ L(mu=0, kappa) p(kappa) dkappa
    """
    kappa_grid_cfg.validate()
    kappa_prior_cfg.validate()

    kappa_grid = jnp.linspace(
        kappa_grid_cfg.kappa_min,
        kappa_grid_cfg.kappa_max,
        kappa_grid_cfg.n_kappa,
    )

    mu0 = 0.0
    logL_kappa = logL_two_obs(
        mu0,
        kappa_grid,
        delta_mu_exp,
        sigma_mu,
        delta_hvp_exp,
        sigma_hvp,
    )
    logPrior_kappa = log_prior_kappa_fn(kappa_grid, kappa_prior_cfg)

    logPost_kappa = logL_kappa + logPrior_kappa

    dkappa = (
        kappa_grid_cfg.kappa_max - kappa_grid_cfg.kappa_min
    ) / (kappa_grid_cfg.n_kappa - 1)
    logZ_SM = jsp.special.logsumexp(logPost_kappa) + jnp.log(dkappa)

    if not jnp.isfinite(logZ_SM):
        print("WARNING: logZ_SM (evidence for SM) is not finite")

    return float(logZ_SM)


# ============================================================
# 6. Wilks toys (two obs) + Brazilian band
# ============================================================

def wilks_pseudo_experiments_two_obs(
    key,
    sigma_mu: float,
    sigma_hvp: float,
    delta_mu_central: float,
    delta_hvp_central: float,
    wilks_cfg: WilksConfig,
):
    """
    Generate Wilks toys for q(mu_true) using the two-observable likelihood.
    """
    wilks_cfg.validate()
    assert sigma_mu > 0.0 and sigma_hvp > 0.0

    mu_cfg = wilks_cfg.mu_grid
    kappa_grid_cfg = wilks_cfg.kappa_grid

    mu_grid = jnp.linspace(mu_cfg.mu_min, mu_cfg.mu_max, mu_cfg.n_mu)
    kappa_grid = jnp.linspace(
        kappa_grid_cfg.kappa_min,
        kappa_grid_cfg.kappa_max,
        kappa_grid_cfg.n_kappa,
    )

    mu_mesh, kappa_mesh = jnp.meshgrid(mu_grid, kappa_grid, indexing="ij")

    def one_toy(key):
        key_eps1, key_eps2 = jax.random.split(key)

        eps1 = jax.random.normal(key_eps1) * sigma_mu
        eps2 = jax.random.normal(key_eps2) * sigma_hvp

        delta_mu_obs = (
            delta_mu_central
            + (wilks_cfg.mu_true + wilks_cfg.kappa_true - delta_mu_central)
            + eps1
        )
        delta_hvp_obs = (
            delta_hvp_central
            + (wilks_cfg.kappa_true - delta_hvp_central)
            + eps2
        )

        logL_2d = logL_two_obs(
            mu_mesh,
            kappa_mesh,
            delta_mu_obs,
            sigma_mu,
            delta_hvp_obs,
            sigma_hvp,
        )
        logL_prof_mu = jnp.max(logL_2d, axis=1)
        logL_max = jnp.max(logL_prof_mu)

        logL_mu_true = jnp.interp(wilks_cfg.mu_true, mu_grid, logL_prof_mu)
        q = -2.0 * (logL_mu_true - logL_max)
        return q

    keys = jax.random.split(key, wilks_cfg.n_toys)
    q_toys = jax.vmap(one_toy)(keys)
    return q_toys


def brazil_band_two_obs(
    key,
    sigma_mu: float,
    sigma_hvp: float,
    delta_mu_central: float,
    delta_hvp_central: float,
    wilks_cfg: WilksConfig,
):
    """
    Brazilian band for q(mu): for each toy, compute q(mu) profile,
    then take median and central intervals at each mu.
    """
    wilks_cfg.validate()
    mu_cfg = wilks_cfg.mu_grid
    kappa_grid_cfg = wilks_cfg.kappa_grid

    mu_grid = jnp.linspace(mu_cfg.mu_min, mu_cfg.mu_max, mu_cfg.n_mu)
    kappa_grid = jnp.linspace(
        kappa_grid_cfg.kappa_min,
        kappa_grid_cfg.kappa_max,
        kappa_grid_cfg.n_kappa,
    )
    mu_mesh, kappa_mesh = jnp.meshgrid(mu_grid, kappa_grid, indexing="ij")

    def one_toy_q_profile(key):
        key_eps1, key_eps2 = jax.random.split(key)
        eps1 = jax.random.normal(key_eps1) * sigma_mu
        eps2 = jax.random.normal(key_eps2) * sigma_hvp

        delta_mu_obs = (
            delta_mu_central
            + (wilks_cfg.mu_true + wilks_cfg.kappa_true - delta_mu_central)
            + eps1
        )
        delta_hvp_obs = (
            delta_hvp_central
            + (wilks_cfg.kappa_true - delta_hvp_central)
            + eps2
        )

        logL_2d = logL_two_obs(
            mu_mesh,
            kappa_mesh,
            delta_mu_obs,
            sigma_mu,
            delta_hvp_obs,
            sigma_hvp,
        )
        logL_prof_mu = jnp.max(logL_2d, axis=1)
        logL_max = jnp.max(logL_prof_mu)
        q_prof_mu = -2.0 * (logL_prof_mu - logL_max)
        return q_prof_mu

    keys = jax.random.split(key, wilks_cfg.n_toys)
    q_profiles = jax.vmap(one_toy_q_profile)(keys)

    q_med = jnp.quantile(q_profiles, 0.5, axis=0)
    q_p16 = jnp.quantile(q_profiles, 0.16, axis=0)
    q_p84 = jnp.quantile(q_profiles, 0.84, axis=0)
    q_p025 = jnp.quantile(q_profiles, 0.025, axis=0)
    q_p975 = jnp.quantile(q_profiles, 0.975, axis=0)

    return {
        "mu_grid": mu_grid,
        "q_med": q_med,
        "q_p16": q_p16,
        "q_p84": q_p84,
        "q_p025": q_p025,
        "q_p975": q_p975,
    }


def wilks_empirical_ci(q_toys, alpha=0.95):
    """
    Empirical quantile of q from toys.
    """
    return jnp.quantile(q_toys, alpha)
# ============================================================
# 7. Prior sweep
# ============================================================

def run_prior_sweep_two_obs(
    delta_mu_exp,
    sigma_mu,
    delta_hvp_exp,
    sigma_hvp,
    mu_cfg,
    kappa_grid_cfg,
    base_prior_cfg,
):
    """
    Compare profile and Bayes mu_hat for different kappa priors.
    """
    prior_specs = [
        ("mixture", base_prior_cfg, log_prior_kappa_mixture_bounded),
        ("uniform", base_prior_cfg, log_prior_kappa_uniform),
        ("gaussian", base_prior_cfg, log_prior_kappa_gaussian),
    ]

    rows = []
    for name, prior_cfg, prior_fn in prior_specs:
        res = profile_and_bayes_two_obs(
            delta_mu_exp=delta_mu_exp,
            sigma_mu=sigma_mu,
            delta_hvp_exp=delta_hvp_exp,
            sigma_hvp=sigma_hvp,
            mu_cfg=mu_cfg,
            kappa_grid_cfg=kappa_grid_cfg,
            kappa_prior_cfg=prior_cfg,
            log_prior_kappa_fn=prior_fn,
        )
        rows.append((name, res["mu_hat_prof"], res["mu_hat_bayes"]))

    print("\n=== Prior sweep (2 obs) ===")
    for name, mu_prof, mu_bayes in rows:
        print(
            f"{name:8s} : mu_hat_profile = {mu_prof:8.2f}, "
            f"mu_hat_bayes = {mu_bayes:8.2f}"
        )
    print("===========================\n")

    return rows


# ============================================================
# 8. MCMC validation (two obs, BlackJAX v1+ style, JIT)
# ============================================================

def run_mcmc_blackjax_two_obs(
    delta_mu_exp,
    sigma_mu,
    delta_hvp_exp,
    sigma_hvp,
    prior_cfg: KappaPriorModel,
    n_samples=3000,
    burn_in=1000,
):
    """
    NUTS sampling of (mu, kappa) posterior using BlackJAX.
    """

    def logpost(theta):
        mu, kappa = theta
        return (
            logL_two_obs(
                mu,
                kappa,
                delta_mu_exp,
                sigma_mu,
                delta_hvp_exp,
                sigma_hvp,
            )
            + log_prior_kappa_mixture_bounded(kappa, prior_cfg)
            + log_prior_mu_wide(mu)
        )

    init = jnp.array([0.0, 50.0])
    step_size = 0.1
    inverse_mass_matrix = jnp.ones(2)

    nuts = blackjax.nuts(
        logpost,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    state = nuts.init(init)

    @jax.jit
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, info

    keys = jax.random.split(jax.random.PRNGKey(123), n_samples + burn_in)

    @jax.jit
    def run_chain(state, keys):
        def body_fn(carry, key):
            s = carry
            s, info = nuts.step(key, s)
            return s, s.position

        final_state, positions = jax.lax.scan(body_fn, state, keys)
        return final_state, positions

    state, positions = run_chain(state, keys)
    samples = positions[burn_in:]

    mu_samps = samples[:, 0]
    kappa_samps = samples[:, 1]

    print("[MCMC] mean(mu)   =", float(jnp.mean(mu_samps)))
    print("[MCMC] mean(kappa)=", float(jnp.mean(kappa_samps)))

    return samples


# ============================================================
# 9. Normalization tests
# ============================================================

def test_prior_normalization(
    prior_cfg: KappaPriorModel,
    prior_fn=log_prior_kappa_mixture_bounded,
):
    """
    Numerical check that the kappa prior integrates to ~1 on its support.
    """
    k = jnp.linspace(prior_cfg.kappa_min, prior_cfg.kappa_max, 20001)
    logp = prior_fn(k, prior_cfg)
    p = jnp.exp(logp)
    dk = k[1] - k[0]
    integral = jnp.sum(p) * dk
    print("Prior normalization:")
    print(f"  ∫p(kappa) dkappa = {float(integral):.6f}")
    print(f"  log(∫p(kappa) dkappa) = {float(jnp.log(integral)):.6f}")


def test_posterior_normalization(res):
    """
    Numerical check that the marginal posterior p(mu) integrates to ~1.
    """
    mu = res["mu_grid"]
    logp = res["logPost_marg_mu"]
    p = jnp.exp(logp)
    dmu = mu[1] - mu[0]
    integral = jnp.sum(p) * dmu
    print("Posterior normalization:")
    print(f"  ∫p(mu) dmu = {float(integral):.6f}")
    print(f"  log(∫p(mu) dmu) = {float(jnp.log(integral)):.6f}")


# ============================================================
# 10. Main: g-2-like toy + Bayes factor + Brazilian band
# ============================================================

def main():
    # g-2-like central values and uncertainties (in 1e-11 units)
    delta_mu_exp = 251.0
    sigma_mu = 59.0

    # Pseudo-HVP constraint (second observable)
    delta_hvp_exp = 70.0
    sigma_hvp = 40.0

    print("delta_mu_exp  (1e-11) =", float(delta_mu_exp))
    print("sigma_mu      (1e-11) =", float(sigma_mu))
    print("delta_hvp_exp (1e-11) =", float(delta_hvp_exp))
    print("sigma_hvp     (1e-11) =", float(sigma_hvp))

    mu_cfg = MuGridConfig(mu_min=-400.0, mu_max=400.0, n_mu=401)

    kappa_prior_cfg = KappaPriorModel(
        w=0.5,
        mu1=0.0,
        sigma1=40.0,
        mu2=150.0,
        sigma2=40.0,
        kappa_min=-100.0,
        kappa_max=300.0,
    )
    kappa_grid_cfg = KappaGridConfig(
        kappa_min=-100.0,
        kappa_max=300.0,
        n_kappa=401,
    )

    # NP hypothesis: mu free
    res_mix = profile_and_bayes_two_obs(
        delta_mu_exp=delta_mu_exp,
        sigma_mu=sigma_mu,
        delta_hvp_exp=delta_hvp_exp,
        sigma_hvp=sigma_hvp,
        mu_cfg=mu_cfg,
        kappa_grid_cfg=kappa_grid_cfg,
        kappa_prior_cfg=kappa_prior_cfg,
        log_prior_kappa_fn=log_prior_kappa_mixture_bounded,
    )

    print("\n[Mixture prior, 2 obs]")
    print("  mu_hat_profile (1e-11) =", res_mix["mu_hat_prof"])
    print("  mu_hat_bayes   (1e-11) =", res_mix["mu_hat_bayes"])
    print("  logZ_NP (mu free)       =", res_mix["logZ"])

    # SM-only hypothesis: mu = 0
    logZ_SM = evidence_sm_two_obs(
        delta_mu_exp=delta_mu_exp,
        sigma_mu=sigma_mu,
        delta_hvp_exp=delta_hvp_exp,
        sigma_hvp=sigma_hvp,
        kappa_grid_cfg=kappa_grid_cfg,
        kappa_prior_cfg=kappa_prior_cfg,
        log_prior_kappa_fn=log_prior_kappa_mixture_bounded,
    )
    print("  logZ_SM (mu=0)          =", logZ_SM)

    logB_NP_SM = res_mix["logZ"] - logZ_SM
    B_NP_SM = jnp.exp(logB_NP_SM)
    print("\n[Bayes factor]")
    print("  log B_NP,SM =", float(logB_NP_SM))
    print("  B_NP,SM     =", float(B_NP_SM))

    run_prior_sweep_two_obs(
        delta_mu_exp,
        sigma_mu,
        delta_hvp_exp,
        sigma_hvp,
        mu_cfg,
        kappa_grid_cfg,
        kappa_prior_cfg,
    )

    print("\n[Normalization tests]")
    test_prior_normalization(
        kappa_prior_cfg,
        log_prior_kappa_mixture_bounded,
    )
    test_posterior_normalization(res_mix)

    mu_grid = res_mix["mu_grid"]
    kappa_grid = res_mix["kappa_grid"]
    mu_mesh, kappa_mesh = res_mix["mu_mesh"], res_mix["kappa_mesh"]

    # 2D posterior: mixture vs uniform prior
    res_unif = profile_and_bayes_two_obs(
        delta_mu_exp=delta_mu_exp,
        sigma_mu=sigma_mu,
        delta_hvp_exp=delta_hvp_exp,
        sigma_hvp=sigma_hvp,
        mu_cfg=mu_cfg,
        kappa_grid_cfg=kappa_grid_cfg,
        kappa_prior_cfg=kappa_prior_cfg,
        log_prior_kappa_fn=log_prior_kappa_uniform,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    logPost_2d_mix = (
        res_mix["logL_2d"]
        + log_prior_kappa_mixture_bounded(kappa_mesh, kappa_prior_cfg)
        + log_prior_mu_wide(mu_mesh)
    )
    ax[0].contourf(
        mu_grid,
        kappa_grid,
        jnp.exp(logPost_2d_mix - jnp.max(logPost_2d_mix)),
        levels=20,
    )
    ax[0].axvline(
        res_mix["mu_hat_prof"],
        color="r",
        ls="--",
        label="profile mu_hat",
    )
    ax[0].axvline(
        res_mix["mu_hat_bayes"],
        color="b",
        ls="--",
        label="Bayes mu_hat",
    )
    ax[0].set_xlabel("mu (1e-11)")
    ax[0].set_ylabel("kappa (1e-11)")
    ax[0].set_title("2D posterior (mixture prior, 2 obs)")
    ax[0].legend()

    logPost_2d_unif = (
        res_unif["logL_2d"]
        + log_prior_kappa_uniform(kappa_mesh, kappa_prior_cfg)
        + log_prior_mu_wide(mu_mesh)
    )
    ax[1].contourf(
        mu_grid,
        kappa_grid,
        jnp.exp(logPost_2d_unif - jnp.max(logPost_2d_unif)),
        levels=20,
    )
    ax[1].axvline(
        res_unif["mu_hat_prof"],
        color="r",
        ls="--",
        label="profile mu_hat",
    )
    ax[1].axvline(
        res_unif["mu_hat_bayes"],
        color="b",
        ls="--",
        label="Bayes mu_hat",
    )
    ax[1].set_xlabel("mu (1e-11)")
    ax[1].set_ylabel("kappa (1e-11)")
    ax[1].set_title("2D posterior (uniform prior, 2 obs)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # 1D profile vs Bayesian posterior in mu
    q_prof_mu = res_mix["q_prof_mu"]
    logPost_marg_mu = res_mix["logPost_marg_mu"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    q95_chi2 = float(chi2_quantile_jax(0.95))

    ax[0].plot(mu_grid, q_prof_mu, label="profile q(mu)")
    ax[0].axhline(
        q95_chi2,
        color="k",
        ls="--",
        lw=1,
        label="95% chi2(df=1)",
    )
    ax[0].set_xlabel("mu (1e-11)")
    ax[0].set_ylabel("q(mu)")
    ax[0].set_title("Profile likelihood (2 obs)")
    ax[0].legend()

    logPost_max = jnp.max(logPost_marg_mu)
    post_q = -2.0 * (logPost_marg_mu - logPost_max)

    ax[1].plot(mu_grid, q_prof_mu, label="profile q(mu)")
    ax[1].plot(
        mu_grid,
        post_q,
        label="Bayes: -2 Δ log p(mu)",
        ls="--",
    )
    ax[1].axhline(
        q95_chi2,
        color="k",
        ls=":",
        lw=1,
        label="95% chi2(df=1)",
    )
    ax[1].set_xlabel("mu (1e-11)")
    ax[1].set_ylabel("-2 Δ log (arb.)")
    ax[1].set_title("Profile vs Bayesian posterior (2 obs)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # Wilks toys: q(mu_true) distribution
    key = jax.random.PRNGKey(0)
    wilks_cfg = WilksConfig(
        mu_true=0.0,
        kappa_true=80.0,
        n_toys=2000,
        mu_grid=MuGridConfig(mu_min=-400.0, mu_max=400.0, n_mu=401),
        kappa_grid=KappaGridConfig(
            kappa_min=-100.0,
            kappa_max=300.0,
            n_kappa=401,
        ),
    )

    q_toys = wilks_pseudo_experiments_two_obs(
        key=key,
        sigma_mu=sigma_mu,
        sigma_hvp=sigma_hvp,
        delta_mu_central=delta_mu_exp,
        delta_hvp_central=delta_hvp_exp,
        wilks_cfg=wilks_cfg,
    )

    mean_q = float(jnp.mean(q_toys))
    std_q = float(jnp.std(q_toys))
    print("\n[Wilks, 2 obs]")
    print("  q_toys mean/std:", mean_q, std_q)

    q_vals = jnp.linspace(0.0, 20.0, 200)

    def emp_cdf_at(q):
        return jnp.mean(q_toys <= q)

    emp_cdf = jax.vmap(emp_cdf_at)(q_vals)
    chi2_cdf = chi2_cdf_jax(q_vals)

    q95_emp = float(wilks_empirical_ci(q_toys, alpha=0.95))
    q95_chi2 = float(chi2_quantile_jax(0.95))
    print("  Empirical 95% Wilks threshold:", q95_emp)
    print("  Asymptotic 95% chi2(df=1) threshold:", q95_chi2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(q_vals, emp_cdf, label="empirical CDF (toys)")
    ax.plot(q_vals, chi2_cdf, label="chi2(df=1) CDF", ls="--")
    ax.axvline(q95_emp, color="r", ls=":", lw=1, label="95% empirical")
    ax.axvline(q95_chi2, color="k", ls="-.", lw=1, label="95% chi2(df=1)")
    ax.set_xlabel("q")
    ax.set_ylabel("CDF")
    ax.set_title("Wilks test: two observables")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Brazilian band: expected q(mu) bands vs observed q(mu)
    key_band = jax.random.PRNGKey(123)
    band = brazil_band_two_obs(
        key=key_band,
        sigma_mu=sigma_mu,
        sigma_hvp=sigma_hvp,
        delta_mu_central=delta_mu_exp,
        delta_hvp_central=delta_hvp_exp,
        wilks_cfg=wilks_cfg,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mu_grid, q_prof_mu, color="k", lw=2, label="observed q(mu)")

    ax.fill_between(
        mu_grid,
        band["q_p16"],
        band["q_p84"],
        color="green",
        alpha=0.4,
        label="expected ±1 sigma",
    )
    ax.fill_between(
        mu_grid,
        band["q_p025"],
        band["q_p975"],
        color="yellow",
        alpha=0.3,
        label="expected ±2 sigma",
    )
    ax.plot(
        mu_grid,
        band["q_med"],
        color="green",
        ls="--",
        lw=1.5,
        label="expected median",
    )

    ax.set_xlabel("mu (1e-11)")
    ax.set_ylabel("q(mu)")
    ax.set_title("Brazilian band for q(mu) (2 obs)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # MCMC validation
    if HAS_BLACKJAX:
        print("\n[MCMC validation, 2 obs]")
        samples = run_mcmc_blackjax_two_obs(
            delta_mu_exp,
            sigma_mu,
            delta_hvp_exp,
            sigma_hvp,
            kappa_prior_cfg,
            n_samples=3000,
            burn_in=1000,
        )

        mu_samps = samples[:, 0]
        kappa_samps = samples[:, 1]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(
            mu_samps,
            bins=50,
            density=True,
            alpha=0.7,
            label="MCMC mu",
        )
        ax[0].plot(
            mu_grid,
            jnp.exp(logPost_marg_mu),
            label="Grid p(mu)",
            lw=1.5,
        )
        ax[0].set_xlabel("mu (1e-11)")
        ax[0].set_ylabel("density")
        ax[0].set_title("mu posterior: MCMC vs grid (2 obs)")
        ax[0].legend()

        ax[1].scatter(mu_samps, kappa_samps, s=2, alpha=0.3)
        ax[1].set_xlabel("mu (1e-11)")
        ax[1].set_ylabel("kappa (1e-11)")
        ax[1].set_title("MCMC samples in (mu, kappa) (2 obs)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
