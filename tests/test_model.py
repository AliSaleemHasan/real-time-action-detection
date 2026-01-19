import os
import sys
import unittest
import tensorflow as tf
from src.models.factory import ModelFactory

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class TestLSTMModel(unittest.TestCase):

    def test_lstm_model_creation(self):
        """Test that ModelFactory creates an LSTM model."""
        config = {
            'architecture': 'LSTM',
            'model': [
                {'layer': 'LSTM', 'units': 32, 'return_sequence': True, 'activation': 'relu'},
                {'layer': 'Dropout', 'drop_perc': 0.2},
                {'layer': 'Dense', 'units': 5, 'activation': 'softmax'}
            ]
        }
        
        factory = ModelFactory.get_model(config)
        model = factory.create_model(input_shape=(30, 51))
        
        self.assertIsInstance(model, tf.keras.models.Sequential)
        self.assertEqual(len(model.layers), 3)
        self.assertIsInstance(model.layers[0], tf.keras.layers.LSTM)

    def test_cnn1d_model_creation(self):
        """Test that ModelFactory creates a CNN1D model."""
        config = {
            'architecture': 'CNN1D',
            'model': [
                {'layer': 'Conv1D', 'filters': 16, 'kernel_size': 3, 'activation': 'relu'},
                {'layer': 'MaxPooling1D', 'pool_size': 2},
                {'layer': 'Flatten'},
                {'layer': 'Dense', 'units': 5, 'activation': 'softmax'}
            ]
        }
        
        from src.models.factory import ModelFactory
        factory = ModelFactory.get_model(config)
        model = factory.create_model(input_shape=(30, 51))
        
        self.assertIsInstance(model, tf.keras.models.Sequential)
        # Layers: Input(Implicit), Conv1D, MaxPooling1D, Flatten, Dense
        self.assertIsInstance(model.layers[0], tf.keras.layers.Conv1D)
        self.assertIsInstance(model.layers[1], tf.keras.layers.MaxPooling1D)

if __name__ == '__main__':
    unittest.main()
