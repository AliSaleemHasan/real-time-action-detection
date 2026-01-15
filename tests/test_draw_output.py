import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.draw_output import draw_features, EDGES

class TestDrawOutput(unittest.TestCase):

    def test_draw_features_runs(self):
        """Test that draw_features runs without error on dummy data."""
        # Create a black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy keypoints: 1 person, 17 keypoints
        # Shape: (1, 17, 3) 
        # We need normalized coordinates (0-1) for x, y
        keypoints = np.random.rand(1, 17, 3)
        # Set scores > 0.5 to ensure drawing happens
        keypoints[0, :, 2] = 0.9 
        
        # Dummy bounding box: 1 box, 5 vals (ymin, xmin, ymax, xmax, score)
        bboxes = np.array([[0.1, 0.1, 0.5, 0.5, 0.9]])
        
        # Should run without error
        try:
            draw_features(frame, keypoints, bboxes, EDGES, threshold=0.5)
        except Exception as e:
            self.fail(f"draw_features raised exception: {e}")
            
        # Check that image was modified (not all zeros anymore)? 
        # Drawing color is often not black, so sum > 0
        self.assertGreater(np.sum(frame), 0)

if __name__ == '__main__':
    unittest.main()
