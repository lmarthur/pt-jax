[![Project Status: Concept – Minimal or no implementation has been done yet, or the repository is only intended to be a limited example, demo, or proof-of-concept.](https://www.repostatus.org/badges/latest/concept.svg)](https://www.repostatus.org/#concept)

![Demo](./plot.jpg)

# pt-jax

Minimal implementation of parallel tempering in JAX.

Parallel tempering is a Markov chain Monte Carlo scheme, designed to sample from complex distributions (e.g., multimodal), for which the standard MCMC samplers may only pseudo-converge.

**Note:** This repository serves *only as a proof of concept* and hosts [a slightly refactored version of the code from this blog post](https://pawel-czyz.github.io/posts/non-reversible-parallel-tempering.html). The goal is to have parallel tempering [supported directly in BlackJAX](https://github.com/blackjax-devs/blackjax/issues/740). However, due to other obligations, I cannot work on finishing the PR until Summer 2025. Hence, before the refactored implementation is added to BlackJAX, perhaps this interim implementation may be useful. After parallel tempering is implemented in BlackJAX, this repository will become read-only.

## Installation

```bash
$ git clone https://github.com/lmarthur/pt-jax.git
$ cd pt-jax
$ pip install -e .
```

## Usage

This package implements two even-odd swap schemes for parallel tempering, both using Metropolis-Hastings acceptance for swap proposals:

- **DEO (Deterministic Even-Odd)**: non-reversible. Parity alternates deterministically with the timestep. Achieves a round-trip rate independent of the number of chains. Described in [S. Syed et al., *Non-Reversible Parallel Tempering: a Scalable Highly Parallel MCMC Scheme* (2019)](https://arxiv.org/abs/1905.02939).
- **SEO (Stochastic Even-Odd)**: reversible. Parity is drawn randomly at each step. Round-trip rate degrades with the number of chains; included for comparison.

See `examples/demo.py` for DEO and `examples/demo_seo.py` for SEO, both using the MALA sampler from BlackJAX. The `pt_jax.annealing` module provides schedule utilities including `annealing_optimal`, which tunes the temperature ladder from empirical rejection rates collected during a pilot run.

## Alternatives

For sampling from complicated distributions, we recommend the following alternatives: 

  - [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl): the state-of-the-art parallel tempering implementation, allowing one to sample using hundreds of machines. If your model is implemented in Julia, this is a great choice.
  - [BlackJAX](https://github.com/blackjax-devs/blackjax/): a wonderful sampling package with many inference methods. For sampling from complicated distributions, we recommend [SMC samplers](https://blackjax-devs.github.io/blackjax/examples/howto_reproduce_the_blackjax_image.html).
  - [TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX): TFP on JAX [supports parallel tempering](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/ReplicaExchangeMC)!
  - [JaxNS](https://github.com/Joshuaalbert/jaxns): nested sampling can approximate many distributions with complicated geometry.
  - [FlowJAX](https://danielward27.github.io/flowjax/): variational inference with normalizing flows [sometimes can outperform MCMC samplers](https://statmodeling.stat.columbia.edu/2024/12/17/applications-of-bayesian-variational-inference/). (See also [this paper](https://arxiv.org/abs/2006.10343) for the review).
