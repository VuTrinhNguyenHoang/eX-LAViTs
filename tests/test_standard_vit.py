import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vit import StandardViT, PatchEmbedding

# Determine the best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running tests on device: {DEVICE}")

class TestPatchEmbedding(unittest.TestCase):
    def setUp(self):
        self.img_size = 32
        self.patch_size = 4
        self.in_chans = 3
        self.embed_dim = 128
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        ).to(DEVICE)

    def test_patch_embedding_initialization(self):
        """Test PatchEmbedding initialization"""
        self.assertEqual(self.patch_embed.num_patches, (self.img_size // self.patch_size) ** 2)
        self.assertIsInstance(self.patch_embed.proj, nn.Conv2d)
        self.assertEqual(self.patch_embed.proj.kernel_size, (self.patch_size, self.patch_size))
        self.assertEqual(self.patch_embed.proj.stride, (self.patch_size, self.patch_size))

    def test_patch_embedding_forward(self):
        """Test PatchEmbedding forward pass"""
        batch_size = 2
        x = torch.randn(batch_size, self.in_chans, self.img_size, self.img_size).to(DEVICE)
        output = self.patch_embed(x)
        
        expected_shape = (batch_size, self.patch_embed.num_patches, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_patch_embedding_different_sizes(self):
        """Test PatchEmbedding with different image sizes"""
        # Test with different image size
        img_size = 64
        patch_size = 8
        patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size).to(DEVICE)
        
        batch_size = 1
        x = torch.randn(batch_size, 3, img_size, img_size).to(DEVICE)
        output = patch_embed(x)
        
        expected_patches = (img_size // patch_size) ** 2
        self.assertEqual(output.shape[1], expected_patches)
        self.assertEqual(output.device.type, DEVICE.type)

class TestStandardViT(unittest.TestCase):
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

    def test_standard_vit_initialization(self):
        """Test StandardViT model initialization"""
        model = StandardViT(**self.default_params).to(DEVICE)
        
        # Check if model is properly initialized
        self.assertIsInstance(model, StandardViT)
        self.assertIsInstance(model.patch_embed, PatchEmbedding)
        self.assertIsInstance(model.cls_token, nn.Parameter)
        self.assertIsInstance(model.pos_embed, nn.Parameter)
        self.assertIsInstance(model.norm, nn.LayerNorm)
        self.assertIsInstance(model.head, nn.Linear)
        
        # Check dimensions
        num_patches = model.patch_embed.num_patches
        self.assertEqual(model.cls_token.shape, (1, 1, self.default_params['embed_dim']))
        self.assertEqual(model.pos_embed.shape, (1, 1 + num_patches, self.default_params['embed_dim']))
        self.assertEqual(model.head.in_features, self.default_params['embed_dim'])
        self.assertEqual(model.head.out_features, self.default_params['num_classes'])
        
        # Check device
        self.assertEqual(model.cls_token.device.type, DEVICE.type)
        self.assertEqual(model.pos_embed.device.type, DEVICE.type)

    def test_standard_vit_forward_pass(self):
        """Test StandardViT forward pass"""
        model = StandardViT(**self.default_params).to(DEVICE)
        model.eval()
        
        x = torch.randn(self.batch_size, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size']).to(DEVICE)
        
        with torch.no_grad():
            output = model(x)
        
        expected_shape = (self.batch_size, self.default_params['num_classes'])
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_standard_vit_gradient_flow(self):
        """Test gradient flow in StandardViT"""
        model = StandardViT(**self.default_params).to(DEVICE)
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

    def test_standard_vit_different_input_sizes(self):
        """Test StandardViT with different input configurations"""
        # Test with different image size
        params = self.default_params.copy()
        params.update({
            'img_size': 64,
            'patch_size': 8,
            'embed_dim': 128
        })
        
        model = StandardViT(**params).to(DEVICE)
        x = torch.randn(1, params['in_chans'], params['img_size'], params['img_size']).to(DEVICE)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, params['num_classes']))
        self.assertEqual(output.device.type, DEVICE.type)

    def test_standard_vit_with_different_params(self):
        """Test StandardViT with various parameter configurations"""
        test_configs = [
            {'num_heads': 8, 'embed_dim': 128},
            {'depth': 6, 'mlp_ratio': 4.0},
            {'drop_rate': 0.1, 'attn_drop_rate': 0.1},
            {'qkv_bias': False, 'act': 'relu'}
        ]
        
        for config in test_configs:
            params = self.default_params.copy()
            params.update(config)
            
            with self.subTest(config=config):
                model = StandardViT(**params).to(DEVICE)
                x = torch.randn(1, params['in_chans'], params['img_size'], params['img_size']).to(DEVICE)
                
                with torch.no_grad():
                    output = model(x)
                
                self.assertEqual(output.shape, (1, params['num_classes']))
                self.assertEqual(output.device.type, DEVICE.type)

    def test_standard_vit_deterministic_output(self):
        """Test that StandardViT produces deterministic output in eval mode"""
        model = StandardViT(**self.default_params).to(DEVICE)
        model.eval()
        
        x = torch.randn(1, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size']).to(DEVICE)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

    def test_standard_vit_parameter_count(self):
        """Test parameter count is reasonable"""
        model = StandardViT(**self.default_params).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertEqual(total_params, trainable_params)
        self.assertGreater(total_params, 0)
        
        # Basic sanity check - should have reasonable number of parameters
        # for the given configuration
        expected_min_params = 1000  # Very conservative lower bound
        self.assertGreater(total_params, expected_min_params)

    def test_standard_vit_device_compatibility(self):
        """Test StandardViT works on different devices"""
        model = StandardViT(**self.default_params)
        x = torch.randn(1, self.default_params['in_chans'], 
                       self.default_params['img_size'], self.default_params['img_size'])
        
        # Test CPU
        model_cpu = model.to('cpu')
        x_cpu = x.to('cpu')
        with torch.no_grad():
            output_cpu = model_cpu(x_cpu)
        
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = x.to('cuda')
            with torch.no_grad():
                output_cuda = model_cuda(x_cuda)
            
            self.assertEqual(output_cuda.device.type, 'cuda')

class TestStandardViTEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_invalid_patch_size(self):
        """Test behavior with edge case patch size"""
        # Patch size equal to image size should work and result in 1 patch
        params = {
            'img_size': 32,
            'patch_size': 32,  # Equal to image size
            'embed_dim': 64,
            'num_heads': 4
        }
        
        # This should not raise an error
        model = StandardViT(**params).to(DEVICE)
        x = torch.randn(1, 3, 32, 32).to(DEVICE)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.device.type, DEVICE.type)

    def test_non_divisible_embed_dim(self):
        """Test error when embed_dim is not divisible by num_heads"""
        with self.assertRaises(AssertionError):
            StandardViT(
                img_size=32,
                patch_size=4,
                embed_dim=65,  # Not divisible by 4
                num_heads=4
            )

    def test_minimum_valid_configuration(self):
        """Test minimum valid configuration"""
        model = StandardViT(
            img_size=4,
            patch_size=4,
            embed_dim=4,
            num_heads=1,
            depth=1
        ).to(DEVICE)
        
        x = torch.randn(1, 3, 4, 4).to(DEVICE)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (1, 1000))  # Default num_classes
        self.assertEqual(output.device.type, DEVICE.type)
