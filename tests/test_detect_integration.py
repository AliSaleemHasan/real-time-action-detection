import unittest
import yaml
import os
import sys


from src.models.factory import ModelFactory

class TestDetectIntegration(unittest.TestCase):
    def test_factory_with_real_config(self):
        """Test that ModelFactory works with the actual config.yaml."""
        config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        factory_model = ModelFactory.get_model(config)
        # Using sequence_length from config
        seq_len = config['sequence_length']
        model = factory_model.create_model(input_shape=(seq_len, 51))
        
        self.assertIsNotNone(model)
        # Check if it matches expected architecture
        if config.get('architecture') == 'CNN1D':
             import tensorflow as tf
             self.assertIsInstance(model.layers[0], tf.keras.layers.Conv1D)

if __name__ == '__main__':
    unittest.main()
