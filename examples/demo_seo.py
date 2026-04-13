"""Demo: SEO (reversible) parallel tempering for sampling from
a two-dimensional distribution employing the MALA kernel from BlackJAX.

SEO uses a stochastic even-odd parity schedule, which is reversible.
Its round-trip rate degrades with the number of chains; it is included
for comparison against the non-reversible DEO variant in demo.py."""
try:
    import matplotlib.pyplot as plt
    import numpyro.distributions as dist
    import blackjax
except ModuleNotFoundError:
    print(
        "Use `pip install matplotlib numpyro` to install additional "
        "packages for this example."
    )

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pt_jax


def log_ref(x):
    return dist.Normal(jnp.zeros(2), 2.0).to_event().log_prob(x)


def log_target(x):
    c0 = jnp.asarray([-2.0, -2.0])
    c1 = jnp.asarray([0.0, 0.0])
    c2 = jnp.asarray([2.0, 2.0])

    t = 50.0

    l0 = -t * (jnp.sum(jnp.square(x - c0)) - 1.0**2) ** 2
    l1 = -t * (jnp.sum(jnp.square(x - c1)) - 1.0**2) ** 2
    l2 = -t * (jnp.sum(jnp.square(x - c2)) - 1.0**2) ** 2

    return jax.nn.logsumexp(jnp.array([l0, l1, l2]))


def mala_kernel_generator(log_p, step_size):
    """Wrapper around the MALA kernel from BlackJAX."""
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, info = mala.step(key, state)
        return new_state.position

    return kernel


def sampling_fn(key, betas, x0, n_samples: int = 5000, warmup: int = 1000):
    n_chains = len(betas)
    step_sizes = jnp.linspace(0.5, 0.01, n_chains)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=mala_kernel_generator,
        params=step_sizes,
    )
    K_seo = pt_jax.swap.generate_seo_extended_kernel(
        log_prob=log_target,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    key, subkey = jrandom.split(key)
    samples, rejection_rates = pt_jax.swap.seo_sampling_loop(
        key=subkey,
        x0=jnp.zeros([n_chains] + list(x0.shape)),
        kernel_local=K_ind,
        kernel_seo=K_seo,
        n_samples=n_samples,
        warmup=warmup,
    )
    # rejection_rates has shape (n_samples, n_chains - 1).
    # Mean over samples gives per-pair rejection rate, which can be passed
    # to pt_jax.annealing.annealing_optimal to tune the temperature ladder.
    mean_rejection_rates = rejection_rates.mean(axis=0)
    return samples, mean_rejection_rates


def main():
    key = jax.random.PRNGKey(2025 - 4 - 7)
    x0 = jnp.zeros(2)

    plot_dims = (4, 5)
    n_chains = plot_dims[0] * plot_dims[1]

    betas = pt_jax.annealing.annealing_exponential(n_chains)

    samples, mean_rejection_rates = sampling_fn(key, betas, x0)

    print("Mean per-pair rejection rates (SEO):")
    for i, r in enumerate(mean_rejection_rates):
        print(f"  chain {i} <-> {i+1}  (beta {betas[i]:.3f} <-> {betas[i+1]:.3f}): {r:.3f}")

    fig, axs = plt.subplots(*plot_dims, sharex=True, sharey=True)

    for i, ax in enumerate(axs.ravel()):
        smp = samples[:, i, :]  # (n_samples, n_chains, n_dims)
        thinning = 5
        ax.scatter(
            smp[::thinning, 0],
            smp[::thinning, 1],
            s=1,
            alpha=0.05,
            c="darkblue",
            rasterized=True,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"$\\beta={betas[i]:.3f}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig("plot_seo.jpg")


if __name__ == "__main__":
    main()
