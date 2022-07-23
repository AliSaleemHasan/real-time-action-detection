'''
    Main Class for DataSet operations including:
    
    1- makeDataSet(model,to,classes,sequence_length,no_sequences,frame_delay,vids_folder)
   
    2- extractFeatures(frame,model):

    3- frames_extraction(video_path,sequence_length )

    4- createDatasetFolders(to,classes)
'''

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os 
import numpy as np
from utilities.draw_output import draw_features

import yaml
from yaml import SafeLoader
import argparse


# get configuration file 
with open('/home/ash/Documents/icdl_detection/config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

parser = argparse.ArgumentParser(description="create dataset from webcam feed or from saved videos on disk")
parser.add_argument("--input",default = None,help="where to collect dataset from \n None for webcam , videoFolderPath for videos ")
args = parser.parse_args()


def extractFeatures(frame,model):
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
    image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 160,192)
    input_img = tf.cast(image, dtype=tf.int32)
    
    # features  is [6,17,57] array of 6 people max , 17 keypoint for each person 
    # each kepoint has (x,y,score) and the rest 5 values are for bounding box   
    features = model(input_img)

    # get keypoints from features
    keypoints = features['output_0'].numpy()[:,:,:51].reshape(6,17,3)

    # get boundingBoxes from features
    bounding_boxes = features['output_0'].numpy()[:,:,51:56].reshape((6,5))

    return keypoints,bounding_boxes


def frames_extraction(video_path,sequence_length = 30):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing  frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list = []
    
    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/sequence_length), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(sequence_length):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video. 
        success, frame = video_reader.read() 

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        
        # Append the normalized frame into the frames list
        frames_list.append(frame)
    
    # Release the VideoCapture object. 
    cv2.destroyAllWindows()
    video_reader.release()

    # Return the frames list.
    return frames_list 


def makeDataSet(model,to,classes,sequence_length=30,no_sequences=1,frame_delay = 1,vids_folder= None):
    '''
        This function will get features from videos (from webcam or exsited videos) as np.array
        and saves it in 'to' folder.
        Args:
            model : used model for feature extraction (multipose lightning in our case)
            to : where to save created dataset 
            classes : what  classes in data set (cheating, notCheating) in our case
            sequence_length: number of extracted frames from each video 
            no_sequence :  how many videos for each action
            frame_delay :EX. if frame_delay is 5 then this funciton will take each frame 
                         that will be divisable by 5  
            vids_folder : if null then video will be capture from webcam , otherwise 
                            videos will be taken from vids_folder 
    '''

    # if vids_floder is not specified then dataSet will be gathered  from webcam
    if vids_folder == None:

        # define capture for webcam 
        cap = cv2.VideoCapture(0)
        

        # loap throw classes to make videos for each action 
        for action in classes :

           # iterate for no_sequence to make no_sequence video for each action
            for sequence in range(no_sequences):


                # iterate for sequence_length time to make sequence_length frame video  
                for frame_num in range(sequence_length):
                
                    # Reading the frame from the video. 
                    success, frame = cap.read() 

                    # Check if Video frame is not successfully read then break the loop
                    if not success:
                        break

                    # extract features from each frame 
                    keypoints, bounding_boxes = extractFeatures(frame,model)

                    draw_features(frame,keypoints,boundingBoxes = bounding_boxes)

                    # add text which indecates starting of new video collection
                    if frame_num == 0:
                        cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV webcam Feed', frame)
                        cv2.waitKey(500)
                    
                    # # add text just to know which video we are reading
                    else:
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV webcam Feed', frame)

                    # specify where to save extracted features
                    npy_path = os.path.join(to, action, str(sequence), str(frame_num))

                    # save extracted features
                    np.save(npy_path, keypoints[0].flatten())

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                       break

        # quite and close camera 
        cap.release()
        cv2.destroyAllWindows()






    # get videos from vids_folder one by one
    else:

        # loap throw all classes in dataset (cheating, notcheating ) in our case 
        for action in classes:

            # loap throw all videos in vids_folder
            for vid_num,video in  enumerate(os.listdir(os.path.join(vids_folder,action))):

                # get required frame list 
                frame_list =frames_extraction(os.path.join(vids_folder,action,str(video)))

                # laop throw returned frame_list to get features from each frame
                for frame_num,frame in enumerate(frame_list):

                    # extract features from frame 
                    keypoints, bounding_boxes = extractFeatures(frame,model)

                    # specify where to save extracted features
                    features_path = os.path.join(to,action,str(vid_num),str(frame_num))

                    # save extracted features in features_path 
                    np.save(features_path,keypoints[0].flatten())
    



def createDatasetFolders(to,classes,no_sequences=30):
    '''
    This function is to create directories which will organaize saved DataSet in directories 

    Args: 
        to: distenation directory
        classes: dataset classes
        no_sequences: number of videos for each action
    
    '''

    # loap throw classes 
    for action in classes :

        # create new folder for each action if not exist
        os.makedirs(os.path.join(to,action),exist_ok=True)

        #iterate no_sequence time to create no_sequence for each action
        for index in range(no_sequences):

            # create no_sequence folders for each action inside action folder 
            os.makedirs(os.path.join(to,action,str(index)),exist_ok=True)

      

def main(config):
    classes = config['classes']
    model_directory = config['model_directory']
    data_directory= "/home/ash/Documents/icdl_detection/DATA_SET"
    sequence_length= config['sequence_length']
    no_sequences = config['no_sequences']
    createDatasetFolders(to=data_directory,classes=classes)
    poseModel = hub.load("/home/ash/Documents/icdl_detection/models/movenet_multipose_lightning_1")
    net = poseModel.signatures['serving_default']
    input = args.input
    makeDataSet(model =net,to =data_directory,classes = classes,sequence_length = sequence_length,no_sequences = no_sequences ,vids_folder=input)


if __name__ == '__main__':
    main(config)






                    

