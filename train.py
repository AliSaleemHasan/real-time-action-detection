''' 
This file is for creating model and train it using tensorFlow 

    Functions :
        LSTM_model(modelConfig)
        getDataSet(actions,datasetPath,sequence_length,test_size)
        Train(model,model_path,X_train,y_train,epochs,optimizer,loss,metric,logsPath)

'''

import yaml
from yaml.loader import SafeLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
import numpy as np
import os 
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical


def LSTM_model(modelConfigPath):
    '''
    This function is for creating our LSTM model 

    Args : 
        modelConfigPath : path for config.yaml file 
    
    Returns :  actionModel (our action detection model)
    '''
    # get all config from config.yaml file
    with open(modelConfigPath) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # get model config from config
    modelConfig = config['model']

    # define new keras Sequential model
    model = Sequential()

    # loop throw each layer in modelConfig
    for index,item in enumerate(modelConfig):

        # if layer is LSTM then add LSTM layer to model
        if item['layer'] == "LSTM":

            # for first LSTM layer we need to add input shape  
            if index == 0:

                # add lstm layer to the model with input shape
                model.add(LSTM(item['units'],return_sequences = item['return_sequence'],activation=item['activation'],input_shape=(30,51)))

            else:

                # add lstm layer to the model without input_shape 
                model.add(LSTM(item['units'],return_sequences = item['return_sequence'],activation=item['activation']))

        # add dense layers to model from modelConfig
        else :
            model.add(Dense(item['units'],activation=item['activation']))
        

    # add last layer 
    model.add(Dense(2,activation="sigmoid"))
        

    return model

def getDataSet(actions,datasetPath,sequence_length,test_size):
    '''
    This function is used to get dataset labels and sequences and split it to 
    input X and output(labels) Y

        Args:   
            actions : our dataset actions
            datasetPath: path for our saved dataset
            sequence_length: number of frame in each video
            test_size: size for testSet
        
        Returns : X_train, X_test, y_train, y_test
    '''

    # intialize list to save all sequences for one video in dataset
    sequences = []

    # intialize list to save label of each video in dataset
    labels = []

    # label map for action (each action points to number)
    labelMap = {label:num for num,label in enumerate(actions)}

    # loop throw action to return sequences and labels for all actions in dataset 
    for action in actions:

        # get all video for action by looping over all videos folders for this action 
        for videoFolder in np.array(os.listdir(os.path.join(datasetPath,action))).astype(int):

            # make window list to save all frame files for one video  
            window = []

            # loop  number of frames times
            for frame_num in range(sequence_length):

                # get frame features that stored as np_array
                res = np.load(os.path.join(datasetPath,action,str(videoFolder),"{}.npy".format(frame_num)))

                # add frame features to window
                window.append(res)

            # add all frame features for one video of one action to sequences list
            sequences.append(window)

            # add label for this video to labels
            labels.append(labelMap[action])

    # create X as np array of sequences
    X = np.array(sequences)

    # y is labels 
    y = to_categorical(labels).astype(int)

    # split DataSet to train test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train,X_test,y_train,y_test






def Train(model,model_path,X_train,y_train,epochs,optimizer,loss,metric,logsPath):
    '''
    This function is to train predefiend model and save training statistic 

        Args:
            model : predefiend model to train 
            model_path: path for previous saved weights if found
            X_train : data to train on 
            y_train : labels of data to train on 
            epoches : number of epochs to train the model 
            optimizer: binary_crossentropy, Adam, RMSpros ...etc
            loss: Probabilistic losses (binary_crossentropy, binary_crossentropy ...etc)
            metric: Accuracy, BinaryAccuracy, BinaryAccuracy, SparseCategoricalAccuracy ...etc
            logsPath: path of tensorflow logs folder 
        
        Return : trained model 

    '''
    
    # if weights are already caclulated then load them to the model and return it 
    if os.path.exists(model_path) == True:

        # load weights into model
        model = model.load_weights(model_path)

        return model

    
    # get logs dir path 
    log_dir = os.path.join(logsPath)

    # define tensorboard callback to save logs in log dir while training 
    tb_callback = TensorBoard(log_dir=log_dir)

    # compile the model
    model.compile(optimizer, loss, metrics=[metric])


    # start training
    model.fit(X_train, y_train, epochs= epochs, callbacks=[tb_callback])

    # save model weights after training
    model.save("models/weights.h5")


    return model





    


