import os
import sys
import unittest
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import LSTM_model

class TestLSTMModel(unittest.TestCase):

    def test_lstm_model_creation(self):
        """Test that LSTM_model creates a Keras model with correct structure from config."""
        
        # Define a dummy model configuration
        model_config = [
            {
                'layer': 'LSTM',
                'units': 32,
                'return_sequence': True,
                'activation': 'relu'
            },
            {
                'layer': 'Dropout',
                'drop_perc': 0.2
            },
            {
                'layer': 'LSTM',
                'units': 16,
                'return_sequence': False,
                'activation': 'relu'
            },
            {
                'layer': 'Dense',
                'units': 5, # 5 classes
                'activation': 'softmax'
            }
        ]

        model = LSTM_model(model_config)

        # Basic assertions
        self.assertIsInstance(model, tf.keras.models.Sequential)
        
        # Check layers
        # Expected: LSTM, Dropout, LSTM, Dense
        # But Sequential might not map 1:1 if input layer is implicit check layers length
        self.assertEqual(len(model.layers), 4)
        
        self.assertIsInstance(model.layers[0], tf.keras.layers.LSTM)
        self.assertEqual(model.layers[0].units, 32)
        
        self.assertIsInstance(model.layers[1], tf.keras.layers.Dropout)
        
        self.assertIsInstance(model.layers[2], tf.keras.layers.LSTM)
        self.assertEqual(model.layers[2].units, 16)
        
        self.assertIsInstance(model.layers[3], tf.keras.layers.Dense)
        self.assertEqual(model.layers[3].units, 5)

if __name__ == '__main__':
    unittest.main()
