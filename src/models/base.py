from abc import ABC, abstractmethod
import tensorflow as tf

class ActionDetectionModel(ABC):
    """
    Abstract base class for action detection models.
    """

    @abstractmethod
    def create_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        Creates and returns a compiled Keras model.
        
        Args:
            input_shape (tuple): The shape of the input data (sequence_length, features).

        Returns:
            tf.keras.Model: The constructed Keras model.
        """
        pass
