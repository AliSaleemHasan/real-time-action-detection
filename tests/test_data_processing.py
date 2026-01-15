import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processing import createDatasetFolders, frames_extraction, deleteNonUsedVids

class TestDataProcessing(unittest.TestCase):

    @patch('utils.data_processing.os.makedirs')
    @patch('utils.data_processing.os.listdir')
    def test_create_dataset_folders(self, mock_listdir, mock_makedirs):
        """Test dataset folder creation."""
        mock_listdir.return_value = ['vid1.mp4', 'vid2.mp4']
        
        createDatasetFolders(
            to='dest_path',
            _from='source_path',
            classes=['action1'],
            augmentation=0,
            no_sequences=2
        )
        
        # Check if makedirs was called for the action and the sequences
        # action1/0, action1/1
        args_list = mock_makedirs.call_args_list
        # We expect at least creation of 'dest_path/action1' and subfolders
        self.assertTrue(mock_makedirs.called)

    @patch('utils.data_processing.cv2.VideoCapture')
    @patch('utils.data_processing.cv2.destroyAllWindows')
    def test_frames_extraction(self, mock_destroy, mock_capture):
        """Test frame extraction from video."""
        # Setup mock video capture
        mock_cap_instance = MagicMock()
        mock_capture.return_value = mock_cap_instance
        
        # Mock frame count and read
        mock_cap_instance.get.return_value = 100 # 100 frames
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, mock_frame)
        
        frames, skip_window = frames_extraction('dummy_path.mp4', sequence_length=10)
        
        self.assertEqual(len(frames), 10)
        self.assertEqual(skip_window, 10) # 100 / 10 = 10
        mock_cap_instance.release.assert_called_once()

    @patch('utils.data_processing.os.remove')
    @patch('utils.data_processing.os.listdir')
    @patch('utils.data_processing.cv2.VideoCapture')
    def test_delete_non_used_vids(self, mock_capture, mock_listdir, mock_remove):
        """Test deletion of videos with insufficient keypoints."""
        mock_listdir.return_value = ['bad_video.mp4']
        
        # Mock video capture
        mock_cap_instance = MagicMock()
        mock_capture.return_value = mock_cap_instance
        mock_cap_instance.get.return_value = 30
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, mock_frame)
        
        # Mock Feature Generator
        mock_fg = MagicMock()
        # Return empty keypoints (score < threshold effectively)
        # extractFeatures returns (keypoints, bbox)
        # keypoints shape expected: (1, 6, 17, 3) or similar based on usage [0]
        # usage: for keypoint in keypoints[0]: if keypoint[2] >= score...
        
        # Create dummy keypoints with low scores
        # Shape should be (NumPeople, NumKeypoints, 3)
        # We simulate 1 person, 17 keypoints
        dummy_kps = np.zeros((1, 17, 3)) 
        dummy_kps[0, :, 2] = 0.1 # all scores < 0.35 default threshold
        
        mock_fg.extractFeatures.return_value = (dummy_kps, None)
        
        mock_model = MagicMock()
        
        deleteNonUsedVids(mock_model, 'dummy_path', sequence_length=5, featureGenerator=mock_fg)
        
        # Should attempt to remove the video because keypoints were insufficient
        mock_remove.assert_called_with(os.path.join('dummy_path', 'bad_video.mp4'))

if __name__ == '__main__':
    unittest.main()
