import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D
from .base import ActionDetectionModel

class CNN1DModel(ActionDetectionModel):
    def __init__(self, config):
        self.config = config

    def create_model(self, input_shape: tuple) -> tf.keras.Model:
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Iterate through config layers
        for item in self.config:
            if item['layer'] == 'Conv1D':
                model.add(Conv1D(filters=item['filters'], kernel_size=item['kernel_size'], activation=item['activation']))
            elif item['layer'] == 'MaxPooling1D':
                model.add(MaxPooling1D(pool_size=item['pool_size']))
            elif item['layer'] == 'Flatten':
                model.add(Flatten())
            elif item['layer'] == 'GlobalMaxPooling1D':
                model.add(GlobalMaxPooling1D())
            elif item['layer'] == 'Dense':
                model.add(Dense(item['units'], activation=item['activation']))
            elif item['layer'] == 'Dropout':
                model.add(Dropout(item['drop_perc']))
                
        return model
