import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import module to test (will use sys.path hack)
try:
    from src.pose_estimation.FeatureGenerator import FeatureGenerator
    from src.pose_estimation.interfaces import PoseEstimator
except ImportError:
    pass

class TestFeatureGenerator(unittest.TestCase):

    def setUp(self):
        self.fg = FeatureGenerator()

    def test_init(self):
        """Test initialization of FeatureGenerator."""
        self.assertEqual(self.fg.keypoints, [])
        self.assertEqual(self.fg.boundingBoxes, [])
        self.assertEqual(self.fg.keypoints_thresh, 0.4)

    def test_extract_features(self):
        """Test extractFeatures method with mocked PoseEstimator."""
        # Setup mocks
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a mock PoseEstimator
        mock_estimator = MagicMock(spec=PoseEstimator)
        
        # Expected output data
        expected_keypoints = np.random.rand(6, 17, 3)
        expected_bboxes = np.random.rand(6, 5)
        
        mock_estimator.estimate.return_value = (expected_keypoints, expected_bboxes)

        # Run extraction
        keypoints, bboxes = self.fg.extractFeatures(mock_frame, mock_estimator)

        # Assertions
        mock_estimator.estimate.assert_called_once_with(mock_frame)
        np.testing.assert_array_equal(keypoints, expected_keypoints)
        np.testing.assert_array_equal(bboxes, expected_bboxes)
        np.testing.assert_array_equal(self.fg.keypoints, expected_keypoints)
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
