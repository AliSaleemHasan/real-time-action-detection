import os
import sys
import unittest
import numpy as np



from utils.tracker import Tracker

class TestTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = Tracker(dist_thresh=0.4, score_thresh=0.4)

    def test_init(self):
        """Test Tracker initialization."""
        self.assertEqual(self.tracker.dist_thresh, 0.4)
        self.assertEqual(self.tracker.score_thresh, 0.4)
        self.assertEqual(self.tracker.id, 0)
        self.assertEqual(len(self.tracker.id2skeleton), 0)

    def test_cost_calculation(self):
        """Test cost calculation between two identical skeletons."""
        # Create a dummy skeleton: 17 keypoints, 3 values (x, y, score)
        skel1 = np.zeros((17, 3))
        # Set scores to > 0.2 to be valid for cost calc
        skel1[:, 2] = 0.5 
        
        # Identical skeletons should have low cost (0 or near 0)
        cost = self.tracker.cost(skel1, skel1)
        self.assertLess(cost, 0.001)

    def test_track_new_person(self):
        """Test tracking a new person."""
        # Single person, 17 keypoints
        curr_skels = np.zeros((1, 17, 3))
        curr_skels[0, :, 2] = 0.9 # High score
        
        curr_bboxes = np.zeros((1, 5)) 
        curr_people = np.array([False] * 6) # Max humans 6
        
        id2skel, id2box, people = self.tracker.track(curr_skels, curr_bboxes, curr_people)
        
        # Should assign an ID (likely 0)
        self.assertIn(0, id2skel)
        self.assertEqual(len(id2skel), 1)

    def test_track_existing_person(self):
        """Test tracking an existing person across frames."""
        # Frame 1
        skel1 = np.zeros((1, 17, 3))
        skel1[0, :, 2] = 0.9
        bbox1 = np.zeros((1, 5))
        people_mask = np.array([False] * 6)
        
        self.tracker.track(skel1, bbox1, people_mask)
        first_id_map = self.tracker.id2skeleton.copy()
        first_id = list(first_id_map.keys())[0]

        # Frame 2: Slight movement
        skel2 = skel1.copy()
        skel2[0, 0, 0] += 0.01 # Small change in x of first keypoint
        
        self.tracker.track(skel2, bbox1, people_mask)
        second_id_map = self.tracker.id2skeleton
        
        # ID should persist
        self.assertIn(first_id, second_id_map)
        self.assertEqual(len(second_id_map), 1)

if __name__ == '__main__':
    unittest.main()
