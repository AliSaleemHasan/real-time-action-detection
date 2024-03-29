'''
handle dataset creation and delete non usable vids in dataset 

'''

import tensorflow_hub as hub
import argparse
import yaml
from yaml import SafeLoader



if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    from utils.data_processing import createDataSet,createDatasetFolders,deleteNonUsedVids
    from utils.FeatureGenerator import FeatureGenerator
    from utils.tracker import Tracker







# get configuration file 
with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

parser = argparse.ArgumentParser(description="create dataset from webcam feed or from saved videos on disk")
parser.add_argument("--input",default = None,help="where to collect dataset from \n None for webcam , videoFolderPath for videos ")

parser.add_argument("--del_nonUsed",help="if true then all non used vids in disered directory will be deleted")

parser.add_argument("--to",default = config['data_directory'],help="if true then all non used vids in disered directory will be deleted")


args = parser.parse_args()






def main(config):
    classes = config['classes']
    model_directory = config['model_directory']
    sequence_length= config['sequence_length']
    no_sequences = config['no_sequences']
    del_nonUsed = args.del_nonUsed
    input = args.input
    to = args.to
    poseModel = hub.load(model_directory)
    net = poseModel.signatures['serving_default']
    fg = FeatureGenerator()
    tracker = Tracker()
        
    if del_nonUsed =="True":
        deleteNonUsedVids(net,input,sequence_length=sequence_length,featureGenerator=fg)
    else:
        createDatasetFolders(to=to,_from=input,classes=classes,augmentation=0,no_sequences=no_sequences)
       
        createDataSet(model =net,to =to,classes = classes,featureGenerator=fg,tracker = tracker,augmentation=0,sequence_length = sequence_length,no_sequences = no_sequences ,vids_folder=input)


if __name__ == '__main__':
    main(config)


