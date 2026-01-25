'''
handle dataset creation and delete non usable vids in dataset 

'''

import tensorflow_hub as hub
import argparse
import yaml
from yaml import SafeLoader




import sys
import os
from utils.data_processing import createDataSet,createDatasetFolders,deleteNonUsedVids
from src.pose_estimation.FeatureGenerator import FeatureGenerator 
from src.pose_estimation.yolo import YOLOPoseEstimator 
from utils.tracker import Tracker
try:
    from utils.gpu_helper import configure_gpu
    configure_gpu()
except ImportError:
    print("Warning: Could not import gpu_helper. Running without explicit GPU configuration.")
from src.config_schema import Config




# get configuration file
with open('config.yaml') as f:
    config_data = yaml.load(f, Loader=SafeLoader)
    config = Config(**config_data)

parser = argparse.ArgumentParser(description="create dataset from webcam feed or from saved videos on disk")
parser.add_argument("--input",default = None,help="where to collect dataset from \n None for webcam , videoFolderPath for videos ")

parser.add_argument("--del_nonUsed",help="if true then all non used vids in disered directory will be deleted")

parser.add_argument("--to",default = config.data_directory,help="if true then all non used vids in disered directory will be deleted")


args = parser.parse_args()






def main(config: Config):
    classes = config.classes
    model_directory = config.model_directory
    sequence_length= config.sequence_length
    no_sequences = config.no_sequences
    del_nonUsed = args.del_nonUsed
    input = args.input
    to = args.to
    
    # Switching to YOLO as default
    pose_estimator = YOLOPoseEstimator(model_path='yolo11n-pose.pt') 
    
    fg = FeatureGenerator()
    tracker = Tracker()

    if del_nonUsed =="True":
        deleteNonUsedVids(pose_estimator,input,sequence_length=sequence_length,featureGenerator=fg)
    else:
        createDatasetFolders(to=to,_from=input,classes=classes,augmentation=0,no_sequences=no_sequences)

        createDataSet(model =pose_estimator,to =to,classes = classes,featureGenerator=fg,tracker = tracker,augmentation=0,sequence_length = sequence_length,no_sequences = no_sequences ,vids_folder=input)


if __name__ == '__main__':
    main(config)


