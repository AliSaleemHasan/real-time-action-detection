'''
This file is used to draw output (keypoints and bounding) boxes in image

    Funcitons:
        draw_keypoints(frame,keypoints,edges,threshold)
        draw_boundingBoxes(frame,boundingBox,theshold)
        draw_connections(frame, keypoints, edges, threshold)
        draw_features(frame,keypoints,edges,boundingBoxes=None,threshold=0.5)
'''
import numpy as np
import cv2 



EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoins(frame,keypoints,threshold):
    '''
    This function is used to draw extraced keypoints from image on it
    
        Args:
            frame: image to draw on in
            keypoints: (6,17,3) array of 6 people , 17 keypoint each consists of (x,y,score)
            threshold: to draw output on image if score of feature is more that it 
        
    '''

    # get frame width,height and channels (RGB)
    y,x,c = frame.shape

    # take keypoints score with respect of frame size
    shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))

    # loop throw keypoints 
    for kp in shaped:

        # get points for each keypoint
        ky, kx, kp_score = kp

        # if score of keypoint is bigger than theshold then we will draw it
        if kp_score > threshold:

            # draw circle for each keypoint that is bigger than theshold 
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)



def draw_boundingBoxes(frame,boundingBox,threshold):
    '''
    This function draws bounding box on image 
    
        Args:
            frame : Image to draw bounding box on 
            boundingBox : (ymin,xmin,ymax,xmax,score) of bounding box 
            threshold : to draw output on image if score of feature is more that it 
        

    '''
    # get frame width,height and channels (RGB)
    y,x,c= frame.shape
    
    # loop throw boundingBox points and score
    for ymin,xmin,ymax,xmax,c in boundingBox:

        # if theshold  is bigger than boundingbox Score then we will take it 
        if  threshold <= c:
            
            #  upper left point of bounding box with respect to image size
            first = (int(xmin * x),int(ymin * y))

            # lower right point of bounding box with respect to image size 
            second = (int(xmax * x),int(ymax * y))

            # draw bounding box 
            cv2.rectangle(frame,first,second,(255,0,0),2)



def draw_connections(frame, keypoints, edges, threshold):

    '''
    This function is to draw connections between each two keypoints

        Args :
            frame : image to draw connections on it
            keypoints :  (6,17,3) array of 6 people , 17 keypoint each consists of (x,y,score) 
            edges : edges between keypints 
            threshold : to draw output on image if score of feature is more that it 

    '''

    # get frame width,height and channels (RGB)
    y, x, c = frame.shape

    # take keypoints score with respect of frame size
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    # loap throw edges
    for edge,color in edges.items():
        
        # take edge coordinate 
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        # if edge points are bigger than threshold then draw line between them
        if (c1 > threshold) & (c2 > threshold):      

            # draw the connection
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


def draw_features(frame,keypoints,boundingBoxes,edges=EDGES,threshold=0.5):
    '''
    This function is used to draw bounding boxes and keypoints on images

        Args:
            frame : image to draw on it 
            keypoints : (6,17,3) array of 6 people , 17 keypoint each consists of (x,y,score)
            boundingBoxes :  (ymin,xmin,ymax,xmax,score) of bounding box 
            edges : edges between keypints 
            threshold : to draw output on image if score of feature is more that it 
    '''
    # loop throw persons in image
    for person in keypoints:
        draw_connections(frame, person, edges, threshold)
        draw_keypoins(frame, person, threshold)

        # drawing bounding box just if we want to 
        if boundingBoxes.size != 0:
            draw_boundingBoxes(frame,boundingBoxes,threshold)
    

