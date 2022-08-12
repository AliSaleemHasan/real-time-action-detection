
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
        augmented_skels = []
        randomness = []

        actual_skels = tracker.actual_skels(self.keypoints)
        if len(actual_skels) == 0:
            return None
        skel = actual_skels[0]
        augmented_skels.append(skel)

        for i in range(output_num):
            randomness.append((np.random.random(skel.shape) - 0.05) * 2 * i * noise_intensity ) 
            
            x=[]
            for j,xi in enumerate(skel):
                if xi[2] > 0.4:
                    xi[:2] +=randomness[i][j][:2]
                x.append(xi)
            
            augmented_skels.append(x)
        
        augmented_skels = np.asarray(augmented_skels)
        return augmented_skels
        
        
        








    
        

        


        
            




