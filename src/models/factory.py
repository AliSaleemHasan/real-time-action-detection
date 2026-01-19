from .lstm import LSTMModel
from .cnn1d import CNN1DModel

class ModelFactory:
    @staticmethod
    def get_model(config_full):
        """
        Returns an instance of an ActionDetectionModel based on the configuration.
        """
        architecture = config_full.get('architecture', 'LSTM') # Default to LSTM
        model_config = config_full['model']
        
        if architecture == 'LSTM':
            return LSTMModel(model_config)
        elif architecture == 'CNN1D':
            return CNN1DModel(model_config)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
