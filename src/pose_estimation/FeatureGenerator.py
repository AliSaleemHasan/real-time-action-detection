'''
    This file is used to 
    1 extract features (keypoints, bounding boxes) from frame 
    2 augment skels (keypoints) to make a bigger dataset
'''


import numpy as np
from .interfaces import PoseEstimator

class FeatureGenerator :

    def __init__(self, keypoints_thresh=0.4) :
        self.keypoints = []
        self.boundingBoxes = []
        self.features = {}
        self.keypoints_thresh = keypoints_thresh


    def extractFeatures(self, frame, pose_estimator: PoseEstimator):
        '''
        This function Extracts Features from image using model 
        Args : 
            frame : image to extract features from 
            pose_estimator : used model to extract features from (PoseEstimator instance)
        Return : 
            keypoints : 17 different key point with (x,y,score) for each one
            bounding_boxes: (ymax,xmax,ymin,xmin,score) 
        '''

        # features  is [6,17,57] array of 6 people max , 17 keypoint for each person 
        # each kepoint has (x,y,score) and the rest 5 values are for bounding box   
        keypoints, bounding_boxes = pose_estimator.estimate(frame)

        self.keypoints = keypoints
        self.boundingBoxes = bounding_boxes
        
        return keypoints, bounding_boxes


    
    def augmentSkels(self,tracker,output_num=4,noise_intensity = 0.1):

        '''
        This function is augmenting skeletons by adding noise to each skeleton by output_num times

        Args:
            tracker:  tracker used to track people in each frame 
            output_num: num of augmented output we want 
            noise_intensity
        '''



        # list of augmented skells 
        augmented_skels = []

        # noise list
        randomness = []

        # get actual skeletons from tracker
        actual_skels = tracker.actual_skels(self.keypoints)

        # if tracker return 0 skels then retrun none
        if len(actual_skels) == 0:
            return None

        # get first skeleton from tracker
        skel = actual_skels[0]

        # add it to augmented skels 
        augmented_skels.append(skel)

        # add each element from random list to base skeleton to make noise each time 
        # then save new skeleton in augmented skeleton
        for i in range(output_num):
            randomness.append((np.random.random(skel.shape) - 0.05) * 2 * i * noise_intensity ) 
            
            x=[]
            for j,xi in enumerate(skel):
                if xi[2] > 0.4:
                    xi[:2] +=randomness[i][j][:2]
                x.append(xi)
            
            augmented_skels.append(x)
        
        # return augmented skels as np array
        augmented_skels = np.asarray(augmented_skels)
        return augmented_skels
