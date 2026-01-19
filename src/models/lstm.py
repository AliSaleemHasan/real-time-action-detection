import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from .base import ActionDetectionModel

class LSTMModel(ActionDetectionModel):
    def __init__(self, config):
        self.config = config

    def create_model(self, input_shape: tuple) -> tf.keras.Model:
        model = Sequential()
        
        # Determine strict structure from config or iterate
        # The original code iterated through a list. We will support that.
        # Assuming config['model'] is the list of layer dicts
        
        model_layers = self.config
        
        for index, item in enumerate(model_layers):
            if item['layer'] == "LSTM":
                if index == 0:
                     model.add(Input(shape=input_shape))
                     model.add(LSTM(item['units'], return_sequences=item['return_sequence'], activation=item['activation']))
                else:
                     model.add(LSTM(item['units'], return_sequences=item['return_sequence'], activation=item['activation']))
            
            elif item['layer'] == 'Dense':
                model.add(Dense(item['units'], activation=item['activation']))
            
            elif item['layer'] == 'Dropout':
                 model.add(Dropout(item['drop_perc']))
        
        return model
