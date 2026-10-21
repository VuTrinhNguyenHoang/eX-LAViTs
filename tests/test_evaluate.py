import unittest
import os
import sys
import tempfile
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import functions to test
from evaluate import (
    load_eval_config, create_args_from_eval_config, create_model_from_config,
    evaluate_model, print_evaluation_results, save_evaluation_results, main
)
from src.models.vit import StandardViT, LAViT

# Determine the best available device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running evaluate tests on device: {DEVICE}")

class TestLoadEvalConfig(unittest.TestCase):
    """Test evaluation configuration loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_eval_config = {
            'model': {
                'checkpoint': '/path/to/checkpoint.pth',
                'config': '/path/to/config.yaml'
            },
            'data': {
                'data_dir': './data',
                'batch_size': 128,
                'num_workers': 4,
                'pin_memory': True
            },
            'output': {
                'output_dir': './output',
                'save_plots': True,
                'save_predictions': False
            },
            'device': {
                'device': 'auto'
            }
        }

    def test_load_eval_config_valid_file(self):
        """Test loading a valid evaluation configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_eval_config, f)
            temp_path = f.name
        
        try:
            loaded_config = load_eval_config(temp_path)
            self.assertEqual(loaded_config, self.test_eval_config)
        finally:
            os.unlink(temp_path)

    def test_load_eval_config_nonexistent_file(self):
        """Test loading a non-existent configuration file"""
        with self.assertRaises(FileNotFoundError):
            load_eval_config('nonexistent_eval_config.yaml')

    def test_load_eval_config_invalid_yaml(self):
        """Test loading an invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                load_eval_config(temp_path)
        finally:
            os.unlink(temp_path)

class TestCreateArgsFromEvalConfig(unittest.TestCase):
    """Test creating args namespace from evaluation config"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.eval_config = {
            'model': {
                'checkpoint': '/path/to/checkpoint.pth',
                'config': '/path/to/config.yaml'
            },
            'data': {
                'data_dir': './data',
                'batch_size': 128,
                'num_workers': 4,
                'pin_memory': True
            },
            'output': {
                'output_dir': './output',
                'save_plots': True,
                'save_predictions': False
            },
            'device': {
                'device': 'auto'
            }
        }

    def test_create_args_complete_eval_config(self):
        """Test creating args from complete evaluation config"""
        args = create_args_from_eval_config(self.eval_config)
        
        # Test model args
        self.assertEqual(args.checkpoint, '/path/to/checkpoint.pth')
        self.assertEqual(args.config, '/path/to/config.yaml')
        
        # Test data args
        self.assertEqual(args.data_dir, './data')
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.num_workers, 4)
        self.assertTrue(args.pin_memory)
        
        # Test output args
        self.assertEqual(args.output_dir, './output')
        self.assertTrue(args.save_plots)
        self.assertFalse(args.save_predictions)
        
        # Test device args
        self.assertEqual(args.device, 'auto')

    def test_create_args_partial_eval_config(self):
        """Test creating args from partial config with defaults"""
        partial_config = {
            'model': {'checkpoint': '/test/checkpoint.pth'}
        }
        args = create_args_from_eval_config(partial_config)
        
        # Test provided values
        self.assertEqual(args.checkpoint, '/test/checkpoint.pth')
        
        # Test default values
        self.assertEqual(args.data_dir, './data')  # default
        self.assertEqual(args.batch_size, 128)  # default
        self.assertEqual(args.num_workers, 4)  # default
        self.assertTrue(args.pin_memory)  # default
        self.assertTrue(args.save_plots)  # default
        self.assertFalse(args.save_predictions)  # default
        self.assertEqual(args.device, 'auto')  # default

    def test_create_args_empty_eval_config(self):
        """Test creating args from empty config"""
        empty_config = {}
        args = create_args_from_eval_config(empty_config)
        
        # Should use all defaults
        self.assertIsNone(args.checkpoint)
        self.assertIsNone(args.config)
        self.assertEqual(args.data_dir, './data')
        self.assertEqual(args.batch_size, 128)

class TestCreateModelFromConfig(unittest.TestCase):
    """Test model creation from configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.standard_config = {
            'model_type': 'StandardViT',
            'img_size': 32,
            'patch_size': 4,
            'num_classes': 10,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'mlp_ratio': 2.0,
            'dropout': 0.1,
            'attention_dropout': 0.0,
            'projection_dropout': 0.0,
            'qkv_bias': True,
            'activation': 'gelu',
            'layer_norm_eps': 1e-6
        }
        
        self.lavit_config = {
            'model_type': 'LAViT',
            'img_size': 32,
            'patch_size': 4,
            'num_classes': 10,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
            'mlp_ratio': 2.0,
            'dropout': 0.1,
            'attention_dropout': 0.0,
            'projection_dropout': 0.0,
            'qkv_bias': True,
            'activation': 'gelu',
            'layer_norm_eps': 1e-6,
            'linear_attention_kernel': 'elu'
        }

    def test_create_standard_vit_from_config(self):
        """Test creating StandardViT model from config"""
        model = create_model_from_config(self.standard_config)
        
        self.assertIsInstance(model, StandardViT)
        # Test forward pass to ensure model is properly created
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_create_lavit_from_config(self):
        """Test creating LAViT model from config"""
        model = create_model_from_config(self.lavit_config)
        
        self.assertIsInstance(model, LAViT)
        # Test forward pass to ensure model is properly created
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_create_model_case_insensitive(self):
        """Test model creation with different case"""
        # Test lowercase
        config_lower = self.standard_config.copy()
        config_lower['model_type'] = 'standardvit'
        model = create_model_from_config(config_lower)
        self.assertIsInstance(model, StandardViT)
        
        # Test mixed case
        config_mixed = self.lavit_config.copy()
        config_mixed['model_type'] = 'LaViT'
        model = create_model_from_config(config_mixed)
        self.assertIsInstance(model, LAViT)

    def test_create_model_invalid_type(self):
        """Test creating model with invalid type"""
        invalid_config = self.standard_config.copy()
        invalid_config['model_type'] = 'InvalidModel'
        
        with self.assertRaises(ValueError) as cm:
            create_model_from_config(invalid_config)
        
        self.assertIn('Unsupported model type', str(cm.exception))

    def test_create_model_default_values(self):
        """Test creating model with default values when config is incomplete"""
        minimal_config = {
            'model_type': 'StandardViT'
        }
        model = create_model_from_config(minimal_config)
        
        # Should use defaults
        self.assertTrue('StandardViT' in str(type(model)))
        # Check num_classes through head layer output features
        self.assertEqual(model.head.out_features, 10)  # CIFAR-10 default

    def test_create_model_parameters(self):
        """Test that model parameters are correctly set"""
        model = create_model_from_config(self.standard_config)
        
        # Check that model was created with correct parameters
        # Check num_classes through head layer output features
        self.assertEqual(model.head.out_features, self.standard_config['num_classes'])
        # Check embed_dim through head layer input features
        self.assertEqual(model.head.in_features, self.standard_config['embed_dim'])

class TestEvaluateModel(unittest.TestCase):
    """Test model evaluation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = StandardViT(
            img_size=32,
            patch_size=4,
            embed_dim=64,
            depth=2,
            num_heads=4,
            num_classes=10
        ).to(DEVICE)
        
        # Create mock test loader
        self.mock_data = [
            (torch.randn(2, 3, 32, 32).to(DEVICE), torch.randint(0, 10, (2,)).to(DEVICE))
        ]
        
        self.class_names = ['class_' + str(i) for i in range(10)]

    def test_evaluate_model_basic(self):
        """Test basic model evaluation"""
        self.model.eval()
        
        # Mock the data loader
        with patch('evaluate.compute_loss_and_accuracy') as mock_compute_loss, \
             patch('evaluate.compute_classification_metrics') as mock_compute_metrics:
            
            # Setup mock returns
            mock_loss_results = {
                'loss': 1.5,
                'top1_accuracy': 75.0,
                'top5_accuracy': 90.0,
                'predictions': np.array([0, 1, 2, 0, 1]),
                'targets': np.array([0, 1, 1, 0, 2])
            }
            mock_compute_loss.return_value = mock_loss_results
            
            mock_classification_results = {
                'macro_precision': 70.0,
                'macro_recall': 65.0,
                'macro_f1': 67.0,
                'weighted_precision': 72.0,
                'weighted_recall': 75.0,
                'weighted_f1': 73.0,
                'confusion_matrix': np.eye(10),
                'classification_report': {}
            }
            mock_compute_metrics.return_value = mock_classification_results
            
            # Run evaluation
            results = evaluate_model(self.model, None, DEVICE, self.class_names)
            
            # Verify results structure
            self.assertIn('loss', results)
            self.assertIn('top1_accuracy', results)
            self.assertIn('top5_accuracy', results)
            self.assertIn('macro_precision', results)
            self.assertIn('macro_recall', results)
            self.assertIn('macro_f1', results)
            self.assertIn('weighted_precision', results)
            self.assertIn('weighted_recall', results)
            self.assertIn('weighted_f1', results)
            self.assertIn('predictions', results)
            self.assertIn('targets', results)
            self.assertIn('confusion_matrix', results)
            self.assertIn('classification_report', results)
            
            # Verify values
            self.assertEqual(results['loss'], 1.5)
            self.assertEqual(results['top1_accuracy'], 75.0)
            self.assertEqual(results['macro_f1'], 67.0)

    def test_evaluate_model_with_real_forward_pass(self):
        """Test evaluation with actual model forward pass (mocked data)"""
        self.model.eval()
        
        # Create a simple test batch
        test_batch = (torch.randn(2, 3, 32, 32).to(DEVICE), 
                     torch.tensor([0, 1]).to(DEVICE))
        
        # Test that model can process the batch
        with torch.no_grad():
            logits = self.model(test_batch[0])
        
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(logits.device.type, DEVICE.type)

class TestPrintEvaluationResults(unittest.TestCase):
    """Test evaluation results printing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_results = {
            'loss': 1.234,
            'top1_accuracy': 85.67,
            'top5_accuracy': 95.32,
            'macro_precision': 83.45,
            'macro_recall': 82.11,
            'macro_f1': 82.77,
            'weighted_precision': 85.90,
            'weighted_recall': 85.67,
            'weighted_f1': 85.78
        }

    def test_print_evaluation_results(self):
        """Test printing evaluation results"""
        # Test that function runs without error
        with patch('builtins.print') as mock_print:
            print_evaluation_results(self.sample_results, "Test Model")
            
            # Verify that print was called
            self.assertTrue(mock_print.called)
            
            # Check that model name is included in output
            printed_output = ''.join([str(call) for call in mock_print.call_args_list])
            self.assertIn("Test Model", printed_output)

    def test_print_evaluation_results_default_name(self):
        """Test printing with default model name"""
        with patch('builtins.print') as mock_print:
            print_evaluation_results(self.sample_results)
            
            printed_output = ''.join([str(call) for call in mock_print.call_args_list])
            self.assertIn("Model", printed_output)

class TestSaveEvaluationResults(unittest.TestCase):
    """Test saving evaluation results"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_results = {
            'loss': 1.234,
            'top1_accuracy': 85.67,
            'top5_accuracy': 95.32,
            'macro_precision': 83.45,
            'macro_recall': 82.11,
            'macro_f1': 82.77,
            'weighted_precision': 85.90,
            'weighted_recall': 85.67,
            'weighted_f1': 85.78,
            'predictions': np.array([0, 1, 2]),  # This should be filtered out
            'targets': np.array([0, 1, 1]),      # This should be filtered out
            'confusion_matrix': np.eye(3),       # This should be filtered out
            'classification_report': {}          # This should be filtered out
        }
        
        self.model_info = {
            'model_type': 'StandardViT',
            'total_params': 1000000,
            'embed_dim': 128
        }

    def test_save_evaluation_results_basic(self):
        """Test saving evaluation results to JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_evaluation_results(self.sample_results, temp_path)
            
            # Read back the results
            with open(temp_path, 'r') as f:
                saved_results = json.load(f)
            
            # Check that serializable results are saved
            self.assertEqual(saved_results['loss'], 1.234)
            self.assertEqual(saved_results['top1_accuracy'], 85.67)
            self.assertEqual(saved_results['macro_f1'], 82.77)
            
            # Check that non-serializable results are not saved
            self.assertNotIn('predictions', saved_results)
            self.assertNotIn('targets', saved_results)
            self.assertNotIn('confusion_matrix', saved_results)
            
        finally:
            os.unlink(temp_path)

    def test_save_evaluation_results_with_model_info(self):
        """Test saving evaluation results with model info"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_evaluation_results(self.sample_results, temp_path, self.model_info)
            
            # Read back the results
            with open(temp_path, 'r') as f:
                saved_results = json.load(f)
            
            # Check that model info is included
            self.assertIn('model_info', saved_results)
            self.assertEqual(saved_results['model_info']['model_type'], 'StandardViT')
            self.assertEqual(saved_results['model_info']['total_params'], 1000000)
            
        finally:
            os.unlink(temp_path)

class TestMainFunction(unittest.TestCase):
    """Test main function (integration test)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_eval_config = {
            'model': {
                'checkpoint': './test_checkpoint.pth',
                'config': None
            },
            'data': {
                'data_dir': './data',
                'batch_size': 8,  # Small batch for testing
                'num_workers': 0,  # No workers for testing
                'pin_memory': False
            },
            'output': {
                'output_dir': None,
                'save_plots': False,  # Don't save plots for testing
                'save_predictions': False
            },
            'device': {
                'device': 'cpu'  # Use CPU for testing
            }
        }

    def test_main_function_config_not_found(self):
        """Test main function with non-existent config"""
        with patch('sys.argv', ['evaluate.py', '--config', 'nonexistent.yaml']):
            with self.assertRaises(FileNotFoundError):
                main()

    def test_main_function_no_checkpoint(self):
        """Test main function with no checkpoint specified"""
        config_without_checkpoint = self.test_eval_config.copy()
        config_without_checkpoint['model']['checkpoint'] = None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_checkpoint, f)
            config_path = f.name
        
        try:
            with patch('sys.argv', ['evaluate.py', '--config', config_path]):
                with self.assertRaises(ValueError) as cm:
                    main()
                
                self.assertIn('Checkpoint path must be provided', str(cm.exception))
        finally:
            os.unlink(config_path)

    @patch('evaluate.get_cifar10_dataloaders')
    @patch('evaluate.load_checkpoint')
    @patch('evaluate.load_config')
    @patch('evaluate.evaluate_model')
    @patch('evaluate.count_parameters')
    @patch('evaluate.get_cifar10_info')
    @patch('evaluate.save_evaluation_results')
    @patch('evaluate.plot_confusion_matrix')
    def test_main_function_successful_run(self, mock_plot_cm, mock_save_results,
                                        mock_get_info, mock_count_params, 
                                        mock_evaluate, mock_load_config, 
                                        mock_load_checkpoint, mock_get_dataloaders):
        """Test successful main function execution (mocked)"""
        
        # Mock CIFAR-10 info
        mock_get_info.return_value = {'classes': ['class_' + str(i) for i in range(10)]}
        
        # Mock data loaders
        mock_test_loader = MagicMock()
        mock_get_dataloaders.return_value = (None, None, mock_test_loader)
        
        # Mock config loading
        mock_model_config = {
            'model_type': 'StandardViT',
            'img_size': 32,
            'embed_dim': 64,
            'num_heads': 4,
            'depth': 2
        }
        mock_load_config.return_value = mock_model_config
        
        # Mock checkpoint loading
        mock_checkpoint = {
            'epoch': 10,
            'metrics': {'val_acc': 85.0}
        }
        mock_load_checkpoint.return_value = mock_checkpoint
        
        # Mock parameter counting
        mock_count_params.return_value = (1000000, 1000000)
        
        # Mock evaluation
        mock_eval_results = {
            'loss': 0.5,
            'top1_accuracy': 80.0,
            'top5_accuracy': 95.0,
            'macro_precision': 78.0,
            'macro_recall': 77.0,
            'macro_f1': 77.5,
            'weighted_precision': 80.5,
            'weighted_recall': 80.0,
            'weighted_f1': 80.2,
            'predictions': np.array([0, 1, 2]),
            'targets': np.array([0, 1, 1]),
            'confusion_matrix': np.eye(10),
            'classification_report': {}
        }
        mock_evaluate.return_value = mock_eval_results
        
        # Mock plotting
        mock_plot_cm.return_value = MagicMock()
        
        # Create temporary config file with valid checkpoint path
        test_config_with_checkpoint = self.test_eval_config.copy()
        test_config_with_checkpoint['model']['checkpoint'] = './test_checkpoint.pth'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config_with_checkpoint, f)
            config_path = f.name
        
        try:
            with patch('sys.argv', ['evaluate.py', '--config', config_path]):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Mock the os.path.exists calls to make it seem like files exist
                    with patch('evaluate.os.path.exists', return_value=True):
                        # Should not raise any errors
                        main()
                        
                        # Verify mocks were called
                        mock_get_dataloaders.assert_called_once()
                        mock_load_checkpoint.assert_called_once()
                        mock_evaluate.assert_called_once()
                    
        finally:
            os.unlink(config_path)

    def test_main_function_checkpoint_override(self):
        """Test main function with checkpoint override via command line"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_eval_config, f)
            config_path = f.name
        
        try:
            with patch('sys.argv', ['evaluate.py', '--config', config_path, '--checkpoint', '/override/checkpoint.pth']):
                # Mock everything to avoid actual file operations
                with patch('evaluate.os.path.exists', return_value=True), \
                     patch('evaluate.load_config', return_value={}), \
                     patch('evaluate.create_model_from_config'), \
                     patch('evaluate.load_checkpoint', return_value={'epoch': 1}), \
                     patch('evaluate.get_cifar10_dataloaders', return_value=(None, None, MagicMock())), \
                     patch('evaluate.get_cifar10_info', return_value={'classes': []}), \
                     patch('evaluate.evaluate_model', return_value={
                         'loss': 0.5, 'top1_accuracy': 80.0, 'top5_accuracy': 95.0,
                         'macro_precision': 78.0, 'macro_recall': 77.0, 'macro_f1': 77.5,
                         'weighted_precision': 80.5, 'weighted_recall': 80.0, 'weighted_f1': 80.2,
                         'predictions': np.array([0, 1, 2]), 'targets': np.array([0, 1, 1]),
                         'confusion_matrix': np.eye(10), 'classification_report': {}
                     }), \
                     patch('evaluate.count_parameters', return_value=(1000, 1000)), \
                     patch('evaluate.save_evaluation_results'), \
                     patch('evaluate.plot_confusion_matrix', return_value=MagicMock()):
                    
                    # Should use the override checkpoint path
                    # This test mainly ensures no errors are raised with override
                    main()
                    
        finally:
            os.unlink(config_path)

class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation components"""
    
    def setUp(self):
        """Set up minimal working configuration"""
        self.minimal_config = {
            'model_type': 'StandardViT',
            'img_size': 32,
            'patch_size': 4,
            'embed_dim': 32,  # Very small for testing
            'depth': 1,       # Minimal depth
            'num_heads': 2,   # Minimal heads
            'num_classes': 10
        }

    def test_end_to_end_config_to_evaluation(self):
        """Test full pipeline from config to model evaluation"""
        # Create model from config
        model = create_model_from_config(self.minimal_config)
        model.eval()
        
        # Test that model can be evaluated (mock the data loader part)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (2, 10))

    def test_model_config_consistency(self):
        """Test that model parameters match config"""
        model = create_model_from_config(self.minimal_config)
        
        # Check key parameters through accessible attributes
        # Check embed_dim through head layer input features
        self.assertEqual(model.head.in_features, self.minimal_config['embed_dim'])
        self.assertEqual(model.head.out_features, self.minimal_config['num_classes'])

if __name__ == '__main__':
    unittest.main()