# WeightForge

[![CI](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Coordinate-based weight generator with selectable output distributions (Fourier, Xavier, Kaiming, Normal, Uniform).

## What it does

WeightForge maps 2D coordinates through a configurable MLP (default 5
layers, 256 wide) to scalar weight values, then applies a distribution
mode: Fourier features, Xavier/Glorot, Kaiming/He, Normal, or Uniform.
Its role in the family is initialization-style generation - matrices
with sensible init statistics, not reconstructions of trained weights.

## Installation

    pip install torch numpy pytest

## Usage

    from weight_forge import create_weight_forge

    forge = create_weight_forge(hidden_size=256, num_layers=5)
    W = forge.forge_weight(1000, 1000)

    forge_k = create_weight_forge(distribution="kaiming")
    Wk = forge_k.forge_weight(100, 100)

## API

- WeightForge(hidden_size=256, num_layers=5, activation="silu",
  distribution="fourier", chunk_size=32768): forge_weight, forge_layer,
  forge_model, count_parameters, compression_ratio
- Activations: silu, relu, gelu, tanh, sigmoid, elu, leaky_relu
- Distributions: fourier, xavier, kaiming, normal, uniform
- WeightForgeWrapper / WeightForgeConfig; create_weight_forge(...) factory

## Honest results and limitations

- The ratio table (up to 10,000,000x) is parameter arithmetic only.
- Generated matrices match the chosen distribution statistically, not
  any trained target. Use this for initialization experiments, not for
  storing or compressing trained models.
- The xavier/kaiming/normal/uniform modes are simple scaled fills; the
  coordinate MLP mostly matters in fourier mode. The non-Fourier modes
  do not need a network at all - using the MLP for them is overhead.
- No accuracy or reconstruction-error numbers are reported because none
  were measured in a controlled experiment.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generated-weight
  reconstructions of trained models were not usable without per-layer
  fitting. The robust storage design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
