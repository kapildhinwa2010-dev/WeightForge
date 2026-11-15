"""
Tests for WeightForge module
"""

import pytest
import torch
import torch.nn as nn
from weight_forge import (
    WeightForge,
    WeightForgeWrapper,
    WeightForgeConfig,
    create_weight_forge,
    WeightDistribution
)


class TestWeightForgeInitialization:
    """Test WeightForge initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        forge = WeightForge()
        assert forge.hidden_size == 256
        assert forge.num_layers == 5
        assert forge.chunk_size == 32768
        assert forge.distribution == "fourier"
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        forge = WeightForge(
            hidden_size=128,
            num_layers=3,
            activation='relu',
            distribution='xavier',
            chunk_size=16384
        )
        assert forge.hidden_size == 128
        assert forge.num_layers == 3
        assert forge.distribution == "xavier"
        assert forge.chunk_size == 16384
    
    def test_network_architecture(self):
        """Test network architecture"""
        forge = WeightForge(hidden_size=64, num_layers=3)
        # Count layers in net
        linear_layers = sum(1 for l in forge.net if isinstance(l, nn.Linear))
        assert linear_layers == 3  # 3 layers (input + 2 hidden + output)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        forge = WeightForge(hidden_size=64, num_layers=3)
        count = forge.count_parameters()
        # 2 input -> 64 + 64 -> 64 + 64 -> 1
        expected = 2*64 + 64 + 64*64 + 64 + 64*1 + 1
        assert count == expected


class TestWeightForgeGeneration:
    """Test weight generation"""
    
    def test_forge_small_matrix(self):
        """Test forging small matrix"""
        forge = WeightForge()
        W = forge.forge_weight(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_forge_square_matrix(self):
        """Test forging square matrix"""
        forge = WeightForge()
        W = forge.forge_weight(100, 100)
        assert W.shape == (100, 100)
    
    def test_forge_rectangular_matrix(self):
        """Test forging rectangular matrix"""
        forge = WeightForge()
        W = forge.forge_weight(200, 100)
        assert W.shape == (200, 100)
    
    def test_forge_matrix_device(self):
        """Test weight generation on CPU"""
        forge = WeightForge()
        W = forge.forge_weight(10, 10, device=torch.device('cpu'))
        assert W.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        forge = WeightForge()
        W = forge.forge_weight(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestWeightForgeChunkedProcessing:
    """Test chunked processing"""
    
    def test_chunked_vs_unchunked(self):
        """Test that chunked and unchunked produce consistent results"""
        forge1 = WeightForge(chunk_size=10000)
        forge2 = WeightForge(chunk_size=100)
        
        # Copy parameters from forge1 to forge2
        forge2.net.load_state_dict(forge1.net.state_dict())
        
        W1 = forge1.forge_weight(100, 100)
        W2 = forge2.forge_weight(100, 100)
        
        # Should be close (may differ slightly due to chunking)
        assert torch.allclose(W1, W2, rtol=1e-4, atol=1e-4)
    
    def test_large_matrix_chunking(self):
        """Test that large matrices are processed correctly"""
        forge = WeightForge(chunk_size=100)
        W = forge.forge_weight(500, 500)
        assert W.shape == (500, 500)
        # Check consistency
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestWeightForgeLayer:
    """Test layer forging"""
    
    def test_forge_layer(self):
        """Test forging multiple layers"""
        forge = WeightForge()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = forge.forge_layer(layer_config)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestWeightForgeModel:
    """Test model forging"""
    
    def test_forge_model(self):
        """Test forging model weights"""
        forge = WeightForge()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        weights = forge.forge_model(model)
        # Should have weights for both linear layers
        assert len(weights) >= 2


class TestWeightForgeWrapper:
    """Test WeightForgeWrapper"""
    
    def test_forge_weights(self):
        """Test forging weights through wrapper"""
        wrapper = WeightForgeWrapper()
        shapes = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = wrapper.forge_weights(shapes)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestCreateWeightForge:
    """Test create_weight_forge factory function"""
    
    def test_default_creation(self):
        """Test default forge creation"""
        forge = create_weight_forge()
        assert isinstance(forge, WeightForge)
        assert forge.hidden_size == 256
        assert forge.num_layers == 5
    
    def test_custom_creation(self):
        """Test custom forge creation"""
        forge = create_weight_forge(
            hidden_size=128,
            num_layers=3,
            activation='gelu',
            distribution='kaiming',
            chunk_size=16384
        )
        assert forge.hidden_size == 128
        assert forge.num_layers == 3
        assert forge.distribution == "kaiming"


class TestWeightForgeConfig:
    """Test WeightForgeConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = WeightForgeConfig()
        assert config.hidden_size == 256
        assert config.num_layers == 5
        assert config.activation == "silu"
        assert config.distribution == "fourier"
        assert config.chunk_size == 32768
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = WeightForgeConfig(
            hidden_size=128,
            num_layers=3,
            activation='relu',
            distribution='xavier'
        )
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.activation == 'relu'
        assert config.distribution == 'xavier'


class TestDistributions:
    """Test different distributions"""
    
    def test_xavier_distribution(self):
        """Test Xavier distribution"""
        forge = WeightForge(distribution='xavier')
        W = forge.forge_weight(100, 100)
        assert W.shape == (100, 100)
    
    def test_kaiming_distribution(self):
        """Test Kaiming distribution"""
        forge = WeightForge(distribution='kaiming')
        W = forge.forge_weight(100, 100)
        assert W.shape == (100, 100)
    
    def test_normal_distribution(self):
        """Test Normal distribution"""
        forge = WeightForge(distribution='normal')
        W = forge.forge_weight(100, 100)
        assert W.shape == (100, 100)
    
    def test_uniform_distribution(self):
        """Test Uniform distribution"""
        forge = WeightForge(distribution='uniform')
        W = forge.forge_weight(100, 100)
        assert W.shape == (100, 100)


class TestActivations:
    """Test different activation functions"""
    
    def test_silu_activation(self):
        """Test SiLU activation"""
        forge = WeightForge(activation='silu')
        W = forge.forge_weight(10, 10)
        assert W.shape == (10, 10)
    
    def test_relu_activation(self):
        """Test ReLU activation"""
        forge = WeightForge(activation='relu')
        W = forge.forge_weight(10, 10)
        assert W.shape == (10, 10)
    
    def test_gelu_activation(self):
        """Test GELU activation"""
        forge = WeightForge(activation='gelu')
        W = forge.forge_weight(10, 10)
        assert W.shape == (10, 10)


class TestCompressionRatio:
    """Test compression ratio calculations"""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        forge = WeightForge(hidden_size=128, num_layers=3)
        
        # Forge has: 2*128 + 128 + 128*128 + 128 + 128*1 + 1 = ~18K params
        forge_params = forge.count_parameters()
        
        # For a 1M parameter model
        original_params = 1_000_000
        ratio = forge.compression_ratio(original_params)
        
        expected_ratio = original_params / forge_params
        assert abs(ratio - expected_ratio) < 0.01


class TestGradientFlow:
    """Test gradient flow through forge"""
    
    def test_gradients_exist(self):
        """Test that gradients flow through forge"""
        forge = WeightForge()
        
        # Forge weights with gradient tracking
        W = forge.forge_weight(10, 10)
        loss = W.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in forge.net.parameters():
            assert param.grad is not None


class TestDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        forge = WeightForge()
        W = forge.forge_weight(10, 10)
        for param in forge.net.parameters():
            assert param.device == W.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        forge = WeightForge()
        W = forge.forge_weight(10, 10)
        for param in forge.net.parameters():
            assert param.dtype == W.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
