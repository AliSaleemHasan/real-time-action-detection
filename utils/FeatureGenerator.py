'''
    This file is used to 
    1 extract features (keypoints, bounding boxes) from frame 
    2 augment skels (keypoints) to make a bigger dataset
'''


import tensorflow as tf
import numpy as np

class FeatureGenerator :

    def __init__(self,keypoints_thresh=0.4) :
        self.keypoints = []
        self.boundingBoxes = []
        self.features = {}
        self.keypoints_thresh = keypoints_thresh


    def extractFeatures(self,frame,model):
        '''
        This function Extracts Features from image using model 
        Args : 
            frame : image to extract features from 
            model : used model to extract features from 
        Return : 
            keypoints : 17 different key point with (x,y,score) for each one
            bounding_boxes: (ymax,xmax,ymin,xmin,score) 
        '''

        # get copy of frame
        image = frame.copy()

        # resize the image (width and hight must be multibles of 32)
        image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0),160 ,256)
        input_img = tf.cast(image, dtype=tf.int32)
        
        # features  is [6,17,57] array of 6 people max , 17 keypoint for each person 
        # each kepoint has (x,y,score) and the rest 5 values are for bounding box   
        features = model(input_img)

        # get keypoints from features
        keypoints = features['output_0'].numpy()[:,:,:51].reshape(6,17,3)

        # get boundingBoxes from features
        bounding_boxes = features['output_0'].numpy()[:,:,51:56].reshape((6,5))

        self.keypoints = keypoints
        self.boundingBoxes = bounding_boxes
        return keypoints,bounding_boxes



    
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
        
        
        








    
        

        


        
            




