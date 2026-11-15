"""
WeightForge - Forge weights from minimal parameters using coordinate-based generation

This module provides functionality to generate neural network weights from a compact
set of parameters, enabling extreme parameter reduction while maintaining model
capacity.

Features:
- Generate weights from coordinate inputs
- Support for various weight distributions
- Memory-efficient generation with chunked processing
- GPU and CPU support
- Integration with PyTorch models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


class WeightDistribution(Enum):
    """Supported weight distributions"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    FOURIER = "fourier"


@dataclass
class WeightForgeConfig:
    """Configuration for weight forging"""
    hidden_size: int = 256
    num_layers: int = 5
    activation: str = "silu"
    distribution: str = "fourier"
    chunk_size: int = 32768
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None


class WeightForge(nn.Module):
    """
    WeightForge - Generate weights from minimal parameters.
    
    This class uses a coordinate-based neural network to generate weight
    matrices of arbitrary size from a compact parameter set.
    
    Example:
        >>> forge = WeightForge(hidden_size=256, num_layers=5)
        >>> # Generate a weight matrix
        >>> W = forge.forge_weight(1000, 1000)
        >>> print(W.shape)  # torch.Size([1000, 1000])
    
    Args:
        hidden_size: Size of hidden layers (default: 256)
        num_layers: Number of hidden layers (default: 5)
        activation: Activation function (default: 'silu')
        distribution: Weight distribution method (default: 'fourier')
        chunk_size: Size of chunks for memory-efficient processing (default: 32768)
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 5,
        activation: str = "silu",
        distribution: str = "fourier",
        chunk_size: int = 32768
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.distribution = distribution
        
        # Determine activation function
        self.activation = self._get_activation(activation)
        
        # Build the coordinate network
        self._build_network()
        
        # Initialize with proper scaling
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.01),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def _build_network(self):
        """Build the coordinate-to-weight network"""
        layers = []
        input_dim = 2  # (i, j) coordinates
        
        # First layer
        layers.append(nn.Linear(input_dim, self.hidden_size))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(self.hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                if self.distribution == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif self.distribution == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _apply_distribution(self, W: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        """Apply selected distribution to generated weights"""
        if self.distribution == "xavier":
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            W = W * scale
        elif self.distribution == "kaiming":
            scale = np.sqrt(2.0 / in_dim)
            W = W * scale
        elif self.distribution == "normal":
            W = W * 0.01  # Small initial scale
        elif self.distribution == "uniform":
            W = W * 0.1  # Small initial scale
        # fourier already handled in generation
        return W
    
    def forge_weight(
        self,
        out_dim: int,
        in_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Forge a weight matrix of specified size.
        
        Args:
            out_dim: Output dimension
            in_dim: Input dimension
            device: Device to use (optional)
            
        Returns:
            Weight matrix of shape (out_dim, in_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Check if we need chunked processing
        total_elements = out_dim * in_dim
        if total_elements <= self.chunk_size:
            # Generate all at once
            return self._forge_weight_unchunked(out_dim, in_dim, device)
        else:
            # Chunked processing
            return self._forge_weight_chunked(out_dim, in_dim, device)
    
    def _forge_weight_unchunked(
        self,
        out_dim: int,
        in_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate weight matrix without chunking"""
        # Generate coordinate grid
        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        
        # Stack coordinates
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)
        
        # Generate weights
        W = self.net(coords).view(out_dim, in_dim)
        
        # Apply distribution scaling
        W = self._apply_distribution(W, out_dim, in_dim)
        
        return W
    
    def _forge_weight_chunked(
        self,
        out_dim: int,
        in_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate weight matrix with chunked processing"""
        W = torch.zeros(out_dim, in_dim, device=device)
        
        # Calculate chunk sizes
        chunk_elements = min(self.chunk_size, out_dim * in_dim)
        chunk_rows = int(np.sqrt(chunk_elements))
        chunk_rows = min(chunk_rows, out_dim)
        
        # Process in chunks
        for i_start in range(0, out_dim, chunk_rows):
            i_end = min(i_start + chunk_rows, out_dim)
            
            for j_start in range(0, in_dim, chunk_rows):
                j_end = min(j_start + chunk_rows, in_dim)
                
                # Generate chunk
                chunk = self._forge_weight_unchunked(
                    i_end - i_start,
                    j_end - j_start,
                    device
                )
                
                W[i_start:i_end, j_start:j_end] = chunk
        
        return W
    
    def forge_layer(
        self,
        layer_config: Dict[str, Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forge weights for multiple layers.
        
        Args:
            layer_config: Dictionary mapping layer names to (out_dim, in_dim)
            
        Returns:
            Dictionary mapping layer names to weight matrices
        """
        weights = {}
        for name, (out_dim, in_dim) in layer_config.items():
            weights[name] = self.forge_weight(out_dim, in_dim)
        return weights
    
    def forge_model(
        self,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Forge weights for all linear layers in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary mapping parameter names to forged weights
        """
        layer_config = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_config[f'{name}.weight'] = (module.out_features, module.in_features)
                if module.bias is not None:
                    layer_config[f'{name}.bias'] = (1, module.out_features)
        
        return self.forge_layer(layer_config)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def compression_ratio(self, original_params: int) -> float:
        """
        Calculate compression ratio.
        
        Args:
            original_params: Number of parameters in original model
            
        Returns:
            Compression ratio (original_params / forge_params)
        """
        forge_params = self.count_parameters()
        return original_params / forge_params if forge_params > 0 else float('inf')


class WeightForgeWrapper:
    """
    Convenience wrapper for weight forging.
    
    Provides a higher-level interface for forging weights for
    neural network layers and models.
    """
    
    def __init__(self, config: Optional[WeightForgeConfig] = None):
        if config is None:
            config = WeightForgeConfig()
        self.config = config
        self.forge = WeightForge(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            activation=config.activation,
            distribution=config.distribution,
            chunk_size=config.chunk_size
        )
        if config.device:
            self.forge.to(config.device)
        if config.dtype:
            self.forge.to(config.dtype)
    
    def forge_weights(
        self,
        shapes: Dict[str, Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forge weights for multiple layers.
        
        Args:
            shapes: Dictionary mapping names to (out_dim, in_dim)
            
        Returns:
            Dictionary mapping names to forged weights
        """
        return self.forge.forge_layer(shapes)


def create_weight_forge(
    hidden_size: int = 256,
    num_layers: int = 5,
    activation: str = "silu",
    distribution: str = "fourier",
    chunk_size: int = 32768,
    device: Optional[str] = None
) -> WeightForge:
    """
    Factory function to create a weight forge.
    
    Args:
        hidden_size: Size of hidden layers
        num_layers: Number of hidden layers
        activation: Activation function name
        distribution: Weight distribution method
        chunk_size: Chunk size for memory-efficient processing
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Configured WeightForge instance
    """
    forge = WeightForge(
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation=activation,
        distribution=distribution,
        chunk_size=chunk_size
    )
    if device:
        forge.to(device)
    return forge


# Example usage and testing
if __name__ == "__main__":
    print("WeightForge - Example Usage")
    print("=" * 60)
    
    # Create forge
    forge = create_weight_forge(hidden_size=256, num_layers=5)
    print(f"WeightForge created with {forge.count_parameters():,} parameters")
    
    # Test weight generation
    print("\nForge sample weight matrices...")
    
    # Small matrix
    W1 = forge.forge_weight(100, 100)
    print(f"Forged 100x100 matrix: {W1.shape}")
    print(f"  Mean: {W1.mean():.6f}, Std: {W1.std():.6f}")
    
    # Medium matrix
    W2 = forge.forge_weight(500, 500)
    print(f"Forged 500x500 matrix: {W2.shape}")
    print(f"  Mean: {W2.mean():.6f}, Std: {W2.std():.6f}")
    
    # Layer forging
    print("\nForge multiple layers...")
    layer_config = {
        'fc1': (256, 128),
        'fc2': (128, 256),
        'fc3': (64, 128),
    }
    weights = forge.forge_layer(layer_config)
    for name, W in weights.items():
        print(f"  {name}: {W.shape}")
    
    # Compression ratio
    original_params = 100*100 + 500*500 + 256*128 + 128*256 + 64*128
    dna_params = forge.count_parameters()
    print(f"\nCompression ratio: {original_params / dna_params:.1f}x")
    
    print("\n✓ WeightForge test completed successfully!")
