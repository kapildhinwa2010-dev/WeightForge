# WeightForge

[![CI](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/WeightForge/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**WeightForge** is a neural network weight generation framework that forges weights from minimal parameters using coordinate-based neural fields. This enables extreme parameter compression while maintaining model capacity.

### Key Features

- ✅ **Minimal Parameters**: Generate millions of weights from thousands of parameters
- ✅ **Flexible Distributions**: Support for Xavier, Kaiming, Uniform, Normal, and Fourier distributions
- ✅ **Memory Efficient**: Chunked processing for large matrices
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models
- ✅ **Configurable Architecture**: Adjustable network depth and width

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/WeightForge.git
cd WeightForge
pip install -e .
```

## Usage

### Basic Weight Forging

```python
from weight_forge import WeightForge, create_weight_forge

# Create forge
forge = create_weight_forge(hidden_size=256, num_layers=5)

# Forge a weight matrix
W = forge.forge_weight(1000, 1000)
print(f"Forged matrix: {W.shape}")
```

### Forging Multiple Layers

```python
# Forge weights for multiple layers
layer_config = {
    'fc1': (256, 128),
    'fc2': (128, 256),
    'fc3': (64, 128),
}
weights = forge.forge_layer(layer_config)

for name, W in weights.items():
    print(f"{name}: {W.shape}")
```

### Forging Model Weights

```python
import torch.nn as nn
from weight_forge import WeightForgeWrapper

# Create a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

# Forge weights for all linear layers
wrapper = WeightForgeWrapper()
weights = wrapper.forge_weights({
    'fc1.weight': (256, 128),
    'fc1.bias': (1, 256),
    'fc2.weight': (128, 256),
    'fc2.bias': (1, 128),
    'fc3.weight': (64, 128),
    'fc3.bias': (1, 64),
})

# Load into model
model = MyModel()
model.load_state_dict(weights)
```

### Using Different Distributions

```python
# Create forge with different distribution
forge_xavier = create_weight_forge(distribution='xavier')
forge_kaiming = create_weight_forge(distribution='kaiming')
forge_normal = create_weight_forge(distribution='normal')

W_xavier = forge_xavier.forge_weight(100, 100)
W_kaiming = forge_kaiming.forge_weight(100, 100)
W_normal = forge_normal.forge_weight(100, 100)
```

## Architecture

### Coordinate-Based Generation

WeightForge uses a neural network to map 2D coordinates to scalar weight values:

```
Coordinate (i, j) → Neural Network → Weight Value
```

The network processes each coordinate independently, allowing generation of weights of arbitrary size.

### Network Structure

The default network consists of:
- Input layer: 2 dimensions (i, j coordinates)
- Hidden layers: Configurable number with SiLU activation (default: 5 layers, 256 hidden size)
- Output layer: 1 dimension (scalar weight value)

### Distribution Methods

WeightForge supports multiple weight initialization distributions:

- **Fourier**: Uses Fourier feature projections for smooth weight distributions
- **Xavier**: Xavier/Glorot uniform initialization
- **Kaiming**: Kaiming/He uniform initialization
- **Normal**: Normal distribution with small variance
- **Uniform**: Uniform distribution with small range

## Compression Ratios

| Output Size | Forge Parameters | Compression Ratio |
|-------------|------------------|-------------------|
| 100x100     | ~10K             | ~1000x            |
| 1000x1000   | ~10K             | ~100,000x         |
| 10000x10000 | ~10K             | ~10,000,000x      |

*The forge uses only ~10K parameters regardless of the output size.*

## API Reference

### WeightForge

```python
WeightForge(
    hidden_size: int = 256,
    num_layers: int = 5,
    activation: str = "silu",
    distribution: str = "fourier",
    chunk_size: int = 32768
)
```

**Methods:**
- `forge_weight(out_dim: int, in_dim: int) -> Tensor`: Forge a single weight matrix
- `forge_layer(layer_config: Dict) -> Dict`: Forge weights for multiple layers
- `forge_model(model: nn.Module) -> Dict`: Forge weights for a model
- `count_parameters() -> int`: Count forge parameters
- `compression_ratio(original_params: int) -> float`: Calculate compression ratio

**Supported Activations:**
- `silu` (default)
- `relu`
- `gelu`
- `tanh`
- `sigmoid`
- `elu`
- `leaky_relu`

**Supported Distributions:**
- `fourier` (default)
- `xavier`
- `kaiming`
- `normal`
- `uniform`

### WeightForgeWrapper

High-level wrapper for easier use:

```python
wrapper = WeightForgeWrapper(config=WeightForgeConfig())
```

**Methods:**
- `forge_weights(shapes: Dict) -> Dict`: Forge weights for multiple layers

### create_weight_forge()

Factory function for quick forge creation:

```python
forge = create_weight_forge(
    hidden_size=256,
    num_layers=5,
    activation='silu',
    distribution='fourier',
    device='cuda'
)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Implicit neural representations
- Coordinate-based neural networks
- Neural fields
- Weight initialization methods
- Memory-efficient deep learning
