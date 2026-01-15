import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import module to test (will use sys.path hack)
try:
    from utils.FeatureGenerator import FeatureGenerator
except ImportError:
    # Handle the case where tensorflow might not be installed in the test env, 
    # though user says they have it. 
    # For robust CI, we might want to mock sys.modules['tensorflow'] before import if needed,
    # but let's assume env is good first.
    pass

class TestFeatureGenerator(unittest.TestCase):

    def setUp(self):
        self.fg = FeatureGenerator()

    def test_init(self):
        """Test initialization of FeatureGenerator."""
        self.assertEqual(self.fg.keypoints, [])
        self.assertEqual(self.fg.boundingBoxes, [])
        self.assertEqual(self.fg.keypoints_thresh, 0.4)

    @patch('utils.FeatureGenerator.tf')
    def test_extract_features(self, mock_tf):
        """Test extractFeatures method with mocked tensorflow operations."""
        # Setup mocks
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the model output
        # Output is expected to be a dictionary with 'output_0' key
        # output_0 shape: [1, 6, 56] (1 image, 6 people, 51 keypoints+5 bbox)
        mock_model_output = {
            'output_0': MagicMock()
        }
        
        # Create dummy output data
        # 6 people, 56 values (17*3 keypoints + 5 bbox)
        dummy_output = np.zeros((1, 6, 56), dtype=np.float32)
        mock_model_output['output_0'].numpy.return_value = dummy_output
        
        mock_model = MagicMock(return_value=mock_model_output)
        
        # Mock tf image resizing and casting to avoid real TF ops
        mock_tf.expand_dims.return_value = mock_frame
        mock_tf.image.resize_with_pad.return_value = mock_frame
        mock_tf.cast.return_value = mock_frame

        # Run extraction
        keypoints, bboxes = self.fg.extractFeatures(mock_frame, mock_model)

        # Assertions
        self.assertEqual(keypoints.shape, (6, 17, 3))
        self.assertEqual(bboxes.shape, (6, 5))
        np.testing.assert_array_equal(self.fg.keypoints, keypoints)
        np.testing.assert_array_equal(self.fg.boundingBoxes, bboxes)

    def test_augment_skels(self):
        """Test augmentSkels with mocked tracker."""
        mock_tracker = MagicMock()
        
        # Mock actual_skels return
        # Shape: (num_skels, 17, 3) 
        input_skel = np.ones((17, 3), dtype=np.float32)
        # Set scores to > 0.4 so augmentation happens
        input_skel[:, 2] = 0.9 
        
        mock_tracker.actual_skels.return_value = [input_skel]
        
        # Set internal keypoints (augmentSkels reads self.keypoints)
        self.fg.keypoints = np.random.rand(6, 17, 3)
        
        augmented = self.fg.augmentSkels(mock_tracker, output_num=2, noise_intensity=0.1)
        
        # Expected output: 1 original + 2 augmented * output_num? 
        # Code reading:
        # skel = actual_skels[0] -> added to list (len=1)
        # loop output_num times:
        #   add new skel to list
        # total = 1 + output_num
        
        self.assertIsNotNone(augmented)
        self.assertEqual(len(augmented), 1 + 2) # 1 original + 2 augmented
        self.assertEqual(augmented[0].shape, (17, 3))

    def test_augment_skels_no_tracker_data(self):
        """Test augmentSkels when tracker returns nothing."""
        mock_tracker = MagicMock()
        mock_tracker.actual_skels.return_value = []
        
        augmented = self.fg.augmentSkels(mock_tracker)
        
        self.assertIsNone(augmented)

if __name__ == '__main__':
    unittest.main()
