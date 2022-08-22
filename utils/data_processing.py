'''
    Main Class for DataSet operations including:
    
    1- createDataSet(model,to,classes,sequence_length,no_sequences,frame_delay,vids_folder)
   
    2- extractFeatures(frame,model):
 
    3- deleteNonUsedVids(model,path,sequence_length,keypoint_score,min_keypoints ):

    3- frames_extraction(video_path,sequence_length )

    4- createDatasetFolders(to,classes)
'''

import cv2
import numpy as np
import os 






def deleteNonUsedVids(model,path,sequence_length,featureGenerator,keypoint_score= 0.35,min_keypoints =2):
    '''
    This function is to remove videos from dataset that does not have 
    enough keypoints to train on it

    Args:
        model: used pose estimation model
        path: path of dataset folder that we want to clean it 
        sequence_length 
        keypoint_score: minimum threshold for keypoint to get considered 
        min_keypoints: minimum number of keypoints to take skeleton
    '''

    # list of videos that not have enough skeletons in it and will be deleted
    to_delete = []
    test_c = 0
    for vid in os.listdir(os.path.join(path)):
        cap = cv2.VideoCapture(os.path.join(path,vid))
        test_c +=1

        # Get the total number of frames in the video.
        video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/sequence_length), 1)




        # skeleton counter in each video 
        skels_counter = 0

        for frame_num in range(sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num * skip_frames_window)
            success , frame = cap.read()

            if not success :
                break 
            keypoints,_ = featureGenerator.extractFeatures(frame,model)



            counter =0
            for keypoint in keypoints[0]:

                # keypoint[2]  is score of keypoint
                # if score of keypoint > score threshold then we will consider it
                if keypoint[2] >= keypoint_score:
                    counter +=1

                # if we considered more than num_of_keypoints then the skeleton is valid
                if counter >=min_keypoints :

                    # save valid skeleton index
                    skels_counter += 1
                    # no need to do this for the rest of keypoints for this skeleton
                    break
            
        # add video into to_delete if number of frames without skels are  equal to sequence length
        if skels_counter  <=  sequence_length -1:
            to_delete.append(os.path.join(path,vid))
    print("there is {} of videos to delete".format(len(to_delete)))
    
    for vid in to_delete:
        os.remove(vid)




        

            

    

def frames_extraction(video_path,sequence_length = 30):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing  frames of the video.

    PS: videos must be same length 
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
    return frames_list , skip_frames_window


def createDataSet(model,to,classes,featureGenerator,tracker,augmentation=1,sequence_length=30,no_sequences=1,vids_folder= None):
    '''
        This function will get features from videos (from webcam or exsited videos) as np.array
        and saves it in 'to' folder.
        Args:
            model : used model for feature extraction (multipose lightning in our case)
            to : where to save created dataset
            extractFeatures: function that extracts feature 
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
        

        # loop throw classes to make videos for each action 
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
                    keypoints, _ = featureGenerator.extractFeatures(frame,model)

                    



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
        
        sequence_rate = 0
        # loop throw all classes in dataset (cheating, notcheating ) in our case 
        for action in classes:
            frame_rate = 0 
            counter =0
            i = - (augmentation + 1)

            # loop throw all videos in vids_folder
            for vid_num,video in  enumerate(os.listdir(os.path.join(vids_folder,action))):
                
                # get required frame list 
                frame_list,skip_frames_window =frames_extraction(os.path.join(vids_folder,action,str(video)))

                frame_rate += skip_frames_window
                counter +=1 

                i += (augmentation + 1)

                
                # laop throw returned frame_list to get features from each frame
                for frame_num,frame in enumerate(frame_list):


                    # extract features from frame 
                    keypoints, _ = featureGenerator.extractFeatures(frame,model)

                    if augmentation != 0:
                        augmentedSkels =featureGenerator.augmentSkels(tracker,augmentation)
                        for j,skel in enumerate(augmentedSkels):
                            # save extracted features in features_path 
                            np.save(os.path.join(to,action,str(i  + j),str(frame_num)),skel.flatten())
                    else :
                        # specify where to save extracted features
                        features_path = os.path.join(to,action,str(vid_num),str(frame_num))
                        np.save(features_path,keypoints[0].flatten())

            rate = max(int(frame_rate / counter),1)
            sequence_rate += rate
        with open('{}/sequence_rate.txt'.format(to),'w') as f:
            f.write(str(int(sequence_rate/len(classes))))

            
            

                
    



def createDatasetFolders(to,_from,classes,augmentation=0,no_sequences=30):
    '''
    This function is to create directories which will organaize saved DataSet in directories 

    Args: 
        to: distenation directory
        classes: dataset classes
        no_sequences: number of videos for each action
        randomness : data augmentation for (randomness) times
    
    '''



    # loop throw classes 
    for action in classes :

        # create new folder for each action if not exist
        os.makedirs(os.path.join(to,action),exist_ok=True)
        if _from != None: 
            no_sequences = len([name for name in os.listdir(os.path.join(_from,action))]) * (augmentation + 1)

        #iterate no_sequence time to create no_sequence for each action
        for index in range(no_sequences):

            # create no_sequence folders for each action inside action folder 
                os.makedirs(os.path.join(to,action,str(index)),exist_ok=True)


      






                    

