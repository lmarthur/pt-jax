import jax
import jax.numpy as jnp
import numpy.testing as npt

import pytest
import pt_jax.kernels as K
import pt_jax.swap as swap


@pytest.mark.parametrize("n_chains", [3, 5])
@pytest.mark.parametrize("shape", [(3,)])
def test_smoke_generate_sample_from_prior_kernel(n_chains: int, shape: tuple):
    def log_p(x):
        return -1.0

    def kernel_generator(log_p, param):
        def k(key, x):
            return x * param

        return k

    params = jnp.linspace(-1, 5, n_chains)
    temperature = jnp.linspace(0, 1, n_chains)

    kernel_ref = kernel_generator(log_p, 1.0)

    kernel = K.generate_sample_from_prior_kernel(
        log_prob=log_p,
        log_ref=log_p,
        kernel_ref=kernel_ref,
        kernel_generator=kernel_generator,
        truncated_annealing_schedule=temperature[1:],
        truncated_params=params[1:],
    )

    x0 = jnp.ones([n_chains] + list(shape))
    key = jax.random.PRNGKey(42)
    x1 = kernel(key, x0)

    assert x1.shape == x0.shape
    # Only works for deterministic kernels
    npt.assert_allclose(x1[0, ...], kernel_ref(key, x0[0, ...]))


@pytest.mark.parametrize("n_chains", [3, 5])
@pytest.mark.parametrize("shape", [(3,)])
def test_smoke_seo_extended_kernel(n_chains: int, shape: tuple):
    def log_prob(x):
        return -1.0

    def log_ref(x):
        return -1.0

    annealing_schedule = jnp.linspace(0.0, 1.0, n_chains)
    kernel = swap.generate_seo_extended_kernel(log_prob, log_ref, annealing_schedule)

    x0 = jnp.ones([n_chains] + list(shape))
    key = jax.random.PRNGKey(42)
    new_state, rejection_rates = kernel(key, x0)

    assert new_state.shape == x0.shape
    assert rejection_rates.shape == (n_chains - 1,)


@pytest.mark.parametrize("n_chains", [3, 5])
@pytest.mark.parametrize("shape", [(3,)])
def test_smoke_seo_sampling_loop(n_chains: int, shape: tuple):
    def log_prob(x):
        return -1.0

    def log_ref(x):
        return -1.0

    annealing_schedule = jnp.linspace(0.0, 1.0, n_chains)

    def kernel_local(key, x):
        return x

    kernel_seo = swap.generate_seo_extended_kernel(log_prob, log_ref, annealing_schedule)

    x0 = jnp.ones([n_chains] + list(shape))
    key = jax.random.PRNGKey(42)
    n_samples = 10
    warmup = 5

    samples, rejection_rates = swap.seo_sampling_loop(
        key=key,
        x0=x0,
        kernel_local=kernel_local,
        kernel_seo=kernel_seo,
        n_samples=n_samples,
        warmup=warmup,
    )

    assert samples.shape == (n_samples, n_chains) + tuple(shape)
    assert rejection_rates.shape == (n_samples, n_chains - 1)
