'''
This file is applying our Action detection on videos (from webcam or saved videos)

    Functions: 
        single_person_detection(skeleton,frame,action_model,sequence_length,actionMap,output_location)
        get_frameSequence(sequence,distance_sequence,frame_num,old_length,skeleton,frame_distance,sequence_length)
        detect(pose_model,action_model,video_path)

'''



import cv2
from sympy import sequence
import tensorflow_hub as hub
from train import LSTM_model
from utilities.data_set import extractFeatures
from utilities.draw_output import draw_features ,EDGES
import numpy as np

from utilities.tracker import Tracker



def single_person_detection(sequence,frame,action_model,sequence_length,actionMap,output_location,id):
    '''
    This function is to perform Action detection for just one person 

        Args:
            sequnce: sequence of skeleton of person to perform Action detection on it 
            frame: image to detect and results on 
            action_model: used LSTM model for action detection
            sequence_lenght: length of skeletons that needed to input them to action_model
            actionMap: dict of action and corresponding label
            output_location: output location on frame 

        Return :
            skel_seq : list of this person human in multiple frames

    '''


    # perform action detection only if there 
    # is {sequence_length} number of skeleton in skel_seq
    if len(sequence) == sequence_length:

        # get results from model
        res = action_model.predict(np.expand_dims(sequence,axis=0))[0]

        # save output action in text variable
        text = actionMap[np.argmax(res)]

        # get frame y and x to put text in correct place according to frame size
        y,x = frame.shape[:2]

        # save predection output with person id to text 
        text += " " + str(id + 1)

        if(output_location[4] > 0.5):
        # put output on frame
          cv2.putText(frame, text, (int(output_location[1] * x),int((output_location[0] * y ) -10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)


        






def get_frameSequence(sequence,distance_sequence,frame_num,old_length,skeleton,frame_distance,sequence_length):
    '''
    This function is to make sure to take the right skeletons to detect on
    EX: first of all when detection started, sequence will be first 30 subsequent frame skeletons,
    but after a while we can make detection on non subsequent frame (each {frame_distance's} frame)
    

        Args : 
            sequence : old sequence
            distance_sequence : old sequence with none subsequent frames
            frame_num: number of current frame
            old_length : old length of frame_distance sequence before adding new frame_distance frame
            skeleton : skeleton to add it to the sequence
            frame_distance : distance between frames that we want to take 
            sequence_length: desired sequence length


        Returns 
            sequence : main_sequence of subsequent frames
            distance_sequence : sequence of non subsequent frames
            old_length : old distance_sequence length if updated 

    ''' 


    # append skeleton to old sequence
    sequence.append(skeleton)



    # take just last 30 skeletons of old sequence
    sequence = sequence[-sequence_length:]

    # check if we reach the desired skeleton to save in distance sequence
    if frame_num % frame_distance == 0 :

        # save desired skeleton on distance sequence
        distance_sequence.append(skeleton)
    
    # get new length of distance_sequence to check if new skeleton is added to it or none
    new_length = len(distance_sequence)
    


    # if new skeleton is added to distance sequence then we will make it as main sequence
    if new_length >= sequence_length and old_length < new_length:

        # check if distance sequence is changed
        old_length = new_length

        # make our main sequence equal to distance sequence
        sequence =  distance_sequence[-sequence_length:]
    

    return sequence,distance_sequence,old_length
    




    


def detect(pose_model,action_model,video_path,actions):
    '''
    This function is to perform action detection on video (saved video or webcam feed)
    by using multiPose detection Model + LSTM model for action detection 

        Args: 
            pose_model: used pose estimation model
            action_model: used LSTM model for action detection
            video_path: video to detect on , or webcam feed if None
            actions: dataset actions

         
    '''
    skels_tracker = Tracker()


    # boolean array of people in image 
    # if value in nth plase is True then there is a person in it
    people = [False] * 6

    # list of person sequence on multiple frames
    skel_seq = [[] for i in range(6)]

    # distance_sequence (taking the nth frames not subseqnet frames)
    frame_sequence = [[] for i in range(6)]

    # old lendth of frame_sequence before adding new skeleton to it
    old_length = 0

    # current frame number
    frame_num = 0

    # dictionary of label and corresponding action
    actionMap = {num:label for num,label in enumerate(actions)}

    # check if detection will be on saved video or webcam feed
    if video_path == None:

        # open webcam feed
        cap = cv2.VideoCapture(0)

    else: 
        cap=cv2.VideoCapture(video_path)


  

    # loop throw webcam feed
    while True:

        # get frame by frame from image feed
        success,frame = cap.read()

        # add frame number
        frame_num += 1

        # if there is error in getting frame then break
        if success == False:
            break

        # extract features from current frame
        keypoints,boundingBoxes = extractFeatures(frame,pose_model)


        # get people dictionary , box dictionary and new_people if added
        dict,box_dict,new_people =skels_tracker.track(keypoints,boundingBoxes,people)


        people = [False] * 6


        for key,value in dict.items():

            # add extracted feature to sequences

            # if one person is added in some id then
            # the corresponding sequence will become empty to start over with this person
            if new_people[key] ==False:
                skel_seq[key]=[]
                frame_sequence[key]=[]

            # get sequences for each person in image
            skel_seq[key],frame_sequence[key],old_length=get_frameSequence(skel_seq[key],frame_sequence[key],frame_num,old_length,value.flatten(),1,30)
            
            # make old_people = new_people
            people[key]= new_people[key]
            
            
            # detect on sequence
            single_person_detection(skel_seq[key],frame,action_model,30,actionMap,box_dict[key],key)

        # draw features (keypoints, boundingBoxes and output of detection if found)
        draw_features(frame,keypoints,boundingBoxes,EDGES)

        # show the image
        cv2.imshow('detect frame',frame)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    
    cap.release()
    cv2.destroyAllWindows()

   



            





            

                
            







pose_model = hub.load("models/movenet_multipose_lightning_1")
net = pose_model.signatures['serving_default']


action_model = LSTM_model('config.yaml')
action_model.load_weights("models/weights.h5")



detect(net,action_model,None,['posing','not_posing'])