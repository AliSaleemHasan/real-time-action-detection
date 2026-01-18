import tensorflow as tf
import numpy as np
from .interfaces import PoseEstimator

class MoveNetPoseEstimator(PoseEstimator):
    """
    Adapter for the existing MoveNet Multipose model using TensorFlow.
    """
    def __init__(self, model_signature):
        """
        Args:
            model_signature: The loaded TensorFlow Hub model signature (e.g. from hub.load(...).signatures['serving_default'])
        """
        self.model = model_signature

    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Logic extracted from original FeatureGenerator.py
        
        # get copy of frame
        image = frame.copy()

        # resize the image (width and hight must be multibles of 32)
        # Original code used resize_with_pad to 160, 256
        image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 160, 256)
        input_img = tf.cast(image, dtype=tf.int32)
        
        # features  is [6,17,57] array of 6 people max , 17 keypoint for each person 
        # each kepoint has (x,y,score) and the rest 5 values are for bounding box   
        features = self.model(input_img)

        # get keypoints from features
        # The original code reshape to (6, 17, 3)
        # Output is assumed to be normalized y, x, score based on Model card, but let's trust existing usage which just passes it through
        keypoints = features['output_0'].numpy()[:,:,:51].reshape(6,17,3)

        # get boundingBoxes from features
        # The original code reshape to (6, 5) -> (ymax, xmax, ymin, xmin, score)
        bounding_boxes = features['output_0'].numpy()[:,:,51:56].reshape((6,5))

        return keypoints, bounding_boxes
