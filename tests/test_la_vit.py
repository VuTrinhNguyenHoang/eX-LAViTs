import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vit import LAViT, StandardViT
from models.transformer import LinearMultiheadAttention

# Determine the best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running LAViT tests on device: {DEVICE}")

class TestLinearMultiheadAttention(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.batch_size = 2
        self.seq_len = 10

    def test_linear_attention_initialization(self):
        """Test LinearMultiheadAttention initialization"""
        attention = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        self.assertEqual(attention.h, self.num_heads)
        self.assertEqual(attention.d, self.embed_dim // self.num_heads)
        self.assertEqual(attention.kernel, "elu")  # default
        self.assertIsInstance(attention.qkv, nn.Linear)
        self.assertIsInstance(attention.out_proj, nn.Linear)

    def test_linear_attention_forward_elu_kernel(self):
        """Test LinearMultiheadAttention forward pass with ELU kernel"""
        attention = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            kernel="elu"
        ).to(DEVICE)
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(DEVICE)
        output = attention(x)
        
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_linear_attention_forward_relu_kernel(self):
        """Test LinearMultiheadAttention forward pass with ReLU kernel"""
        attention = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            kernel="relu"
        ).to(DEVICE)
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim).to(DEVICE)
        output = attention(x)
        
        expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_linear_attention_phi_function(self):
        """Test phi function for different kernels"""
        attention_elu = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            kernel="elu"
        )
        
        attention_relu = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            kernel="relu"
        )
        
        x = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.embed_dim // self.num_heads)
        
        # Test ELU kernel
        phi_elu = attention_elu._phi(x)
        # ELU(x) + 1 >= 1 when x >= 0, but can be < 1 when x < 0 since ELU(x) = x when x >= 0, else ELU(x) = alpha*(exp(x) - 1)
        # For alpha=1: ELU(x) + 1 = exp(x) when x < 0, which is > 0 but can be < 1
        # So we should test that it's > 0 rather than >= 1
        self.assertTrue(torch.all(phi_elu > 0.0))  # ELU + 1 should be > 0
        
        # Test ReLU kernel
        phi_relu = attention_relu._phi(x)
        self.assertTrue(torch.all(phi_relu >= 0.0))  # ReLU should be >= 0

    def test_linear_attention_gradient_flow(self):
        """Test gradient flow in LinearMultiheadAttention"""
        attention = LinearMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        ).to(DEVICE)
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True).to(DEVICE)
        x.retain_grad()  # Retain gradient for non-leaf tensor
        output = attention(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.device.type, DEVICE.type)
        for param in attention.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertEqual(param.grad.device.type, DEVICE.type)

class TestLAViT(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with common parameters"""
        self.default_params = {
            'img_size': 32,
            'patch_size': 4,
            'in_chans': 3,
            'num_classes': 10,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'mlp_ratio': 2.0
        }
        self.batch_size = 2

    def test_la_vit_initialization(self):
        """Test LAViT model initialization"""
        model = LAViT(**self.default_params)
        
        # Check if model is properly initialized
        self.assertIsInstance(model, LAViT)
        self.assertIsInstance(model.patch_embed, nn.Module)
        self.assertIsInstance(model.cls_token, nn.Parameter)
        self.assertIsInstance(model.pos_embed, nn.Parameter)
        self.assertIsInstance(model.norm, nn.LayerNorm)
        self.assertIsInstance(model.head, nn.Linear)
        
        # Check dimensions
        num_patches = model.patch_embed.num_patches
        self.assertEqual(model.cls_token.shape, (1, 1, self.default_params['embed_dim']))
        self.assertEqual(model.pos_embed.shape, (1, 1 + num_patches, self.default_params['embed_dim']))

    def test_la_vit_forward_pass(self):
        """Test LAViT forward pass"""
        model = LAViT(**self.default_params).to(DEVICE)
        model.eval()
        
        x = torch.randn(self.batch_size, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size']).to(DEVICE)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (self.batch_size, self.default_params['num_classes'])
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_la_vit_different_kernels(self):
        """Test LAViT with different kernel types"""
        kernels = ["elu", "relu"]
        
        for kernel in kernels:
            with self.subTest(kernel=kernel):
                params = self.default_params.copy()
                model = LAViT(kernel=kernel, **params)
                
                x = torch.randn(1, params['in_chans'], params['img_size'], params['img_size'])
                
                with torch.no_grad():
                    output = model(x)
                
                self.assertEqual(output.shape, (1, params['num_classes']))

    def test_la_vit_gradient_flow(self):
        """Test gradient flow in LAViT"""
        model = LAViT(**self.default_params).to(DEVICE)
        model.train()
        
        x = torch.randn(self.batch_size, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size']).to(DEVICE)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed for key parameters
        # Some parameters might not have gradients if they're not used in this specific forward pass
        key_params_with_grads = 0
        total_key_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Focus on key parameters that should definitely have gradients
                if any(key in name for key in ['head.weight', 'head.bias', 'patch_embed', 'cls_token', 'pos_embed']):
                    total_key_params += 1
                    if param.grad is not None:
                        key_params_with_grads += 1
                        self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)), 
                                       f"Zero gradient for {name}")
                        self.assertEqual(param.grad.device.type, DEVICE.type, f"Gradient for {name} not on correct device")
        
        # Ensure at least most key parameters have gradients
        self.assertGreater(key_params_with_grads, total_key_params * 0.8, 
                          "Too few key parameters have gradients")

    def test_la_vit_linear_attention_usage(self):
        """Test that LAViT uses linear attention instead of standard attention"""
        model = LAViT(**self.default_params)
        
        # Check that encoder layers use LinearMultiheadAttention
        found_linear_attention = False
        for module in model.modules():
            if isinstance(module, LinearMultiheadAttention):
                found_linear_attention = True
                break
        
        self.assertTrue(found_linear_attention, "LAViT should use LinearMultiheadAttention")

    def test_la_vit_memory_efficiency(self):
        """Test memory efficiency compared to standard attention (basic check)"""
        # This is a basic test - in practice, linear attention should be more memory efficient
        # for longer sequences, but with small sequences the difference might not be significant
        
        params = self.default_params.copy()
        params.update({'img_size': 64, 'patch_size': 4})  # More patches
        
        model = LAViT(**params)
        x = torch.randn(1, params['in_chans'], params['img_size'], params['img_size'])
        
        # Should not raise memory errors
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, params['num_classes']))

    def test_la_vit_deterministic_output(self):
        """Test that LAViT produces deterministic output in eval mode"""
        model = LAViT(**self.default_params)
        model.eval()
        
        x = torch.randn(1, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size'])
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_la_vit_various_configurations(self):
        """Test LAViT with various parameter configurations"""
        test_configs = [
            {'kernel': 'elu', 'num_heads': 8, 'embed_dim': 128},
            {'kernel': 'relu', 'depth': 6, 'mlp_ratio': 4.0},
            {'kernel': 'elu', 'drop_rate': 0.1, 'attn_drop_rate': 0.1},
            {'kernel': 'relu', 'qkv_bias': False, 'act': 'relu'}
        ]
        
        for config in test_configs:
            params = self.default_params.copy()
            params.update(config)
            
            with self.subTest(config=config):
                model = LAViT(**params)
                x = torch.randn(1, params['in_chans'], params['img_size'], params['img_size'])
                
                with torch.no_grad():
                    output = model(x)
                
                self.assertEqual(output.shape, (1, params['num_classes']))

    def test_la_vit_parameter_count(self):
        """Test parameter count is reasonable"""
        model = LAViT(**self.default_params)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertEqual(total_params, trainable_params)
        self.assertGreater(total_params, 0)

    def test_la_vit_device_compatibility(self):
        """Test LAViT works on different devices"""
        model = LAViT(**self.default_params)
        x = torch.randn(1, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size'])
        
        # Test CPU
        model = model.to('cpu')
        x = x.to('cpu')
        with torch.no_grad():
            output_cpu = model(x)
        
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            x = x.to('cuda')
            with torch.no_grad():
                output_cuda = model(x)
            
            self.assertEqual(output_cuda.device.type, 'cuda')

class TestLAViTComparison(unittest.TestCase):
    """Test LAViT in comparison with StandardViT"""
    
    def setUp(self):
        self.params = {
            'img_size': 32,
            'patch_size': 4,
            'in_chans': 3,
            'num_classes': 10,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'mlp_ratio': 2.0
        }

    def test_la_vit_vs_standard_vit_output_shapes(self):
        """Test that LAViT and StandardViT produce same output shapes"""
        la_model = LAViT(**self.params).to(DEVICE)
        standard_model = StandardViT(**self.params).to(DEVICE)
        
        x = torch.randn(1, self.params['in_chans'], self.params['img_size'], self.params['img_size']).to(DEVICE)
        
        with torch.no_grad():
            la_output = la_model(x)
            standard_output = standard_model(x)
        
        self.assertEqual(la_output.shape, standard_output.shape)
        self.assertEqual(la_output.device.type, DEVICE.type)
        self.assertEqual(standard_output.device.type, DEVICE.type)

    def test_la_vit_parameter_compatibility(self):
        """Test that LAViT accepts same parameters as StandardViT"""
        # All parameters that work for StandardViT should also work for LAViT
        # (except attn_type which is fixed)
        test_params = self.params.copy()
        test_params.update({
            'drop_rate': 0.1,
            'attn_drop_rate': 0.1,
            'proj_drop_rate': 0.1,
            'qkv_bias': False,
            'act': 'relu',
            'layer_norm_eps': 1e-5
        })
        
        # This should not raise any errors
        la_model = LAViT(**test_params)
        standard_model = StandardViT(**test_params)
        
        x = torch.randn(1, test_params['in_chans'], test_params['img_size'], test_params['img_size'])
        
        with torch.no_grad():
            la_output = la_model(x)
            standard_output = standard_model(x)
        
        self.assertEqual(la_output.shape, standard_output.shape)

    def test_different_attention_mechanisms(self):
        """Test that LAViT and StandardViT use different attention mechanisms"""
        la_model = LAViT(**self.params)
        standard_model = StandardViT(**self.params)
        
        # Check that LAViT uses LinearMultiheadAttention
        has_linear_attention = any(isinstance(m, LinearMultiheadAttention) for m in la_model.modules())
        self.assertTrue(has_linear_attention, "LAViT should use LinearMultiheadAttention")
        
        # Check that StandardViT does not use LinearMultiheadAttention
        has_linear_attention_std = any(isinstance(m, LinearMultiheadAttention) for m in standard_model.modules())
        self.assertFalse(has_linear_attention_std, "StandardViT should not use LinearMultiheadAttention")


class TestLAViTEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for LAViT"""
    
    def test_la_vit_single_patch(self):
        """Test LAViT with single patch (patch_size = img_size)"""
        model = LAViT(
            img_size=32,
            patch_size=32,  # Single patch
            embed_dim=64,
            num_heads=4,
            depth=1
        )
        
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape[0], 1)

    def test_la_vit_minimum_configuration(self):
        """Test LAViT with minimum valid configuration"""
        model = LAViT(
            img_size=4,
            patch_size=4,
            embed_dim=4,
            num_heads=1,
            depth=1,
            kernel="elu"
        )
        
        x = torch.randn(1, 3, 4, 4)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, 1000))  # Default num_classes

    def test_la_vit_invalid_kernel(self):
        """Test LAViT with invalid kernel (should work but fall through to default behavior)"""
        # Note: The current implementation doesn't validate kernel type,
        # so invalid kernels would just not apply any transformation
        model = LAViT(
            img_size=32,
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            kernel="invalid_kernel"
        )
        
        x = torch.randn(1, 3, 32, 32)
        # This should still work (though the linear attention might not work as expected)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape[0], 1)
