''' 
This file is for creating model and train it using tensorFlow 

    Functions :
        LSTM_model(modelConfig)
        getDataSet(classes,datasetPath,sequence_length,test_size)
        Train(model,model_path,X_train,y_train,epochs,optimizer,loss,metric,logsPath)

'''

import os 
import yaml
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.metrics import BinaryAccuracy,Accuracy
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix



logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def LSTM_model(modelConfig):
    '''
    This function is for creating our LSTM model 

    Args : 
        modelConfigPath : model configuration  
    
    Returns :  actionModel (our action detection model)
    '''
 

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

def getDataSet(classes,datasetPath,sequence_length,test_size):
    '''
    This function is used to get dataset labels and sequences and split it to 
    input X and output(labels) Y

        Args:   
            classes : our dataset classes
            datasetPath: path for our saved dataset
            sequence_length: number of frame in each video
            test_size: size for testSet
        
        Returns : X_train, X_test, y_train, y_test
    '''
    start_time = time.perf_counter()

    # intialize list to save all sequences for one video in dataset
    sequences = []

    # intialize list to save label of each video in dataset
    labels = []

    # label map for action (each action points to number)
    labelMap = {label:num for num,label in enumerate(classes)}

    # loop throw action to return sequences and labels for all classes in dataset 
    for action in classes:

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

    end_time = time.perf_counter()
    logging.info(f'It took {end_time- start_time :0.2f} second(s) to complete.')


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
        model.load_weights(model_path)

        model.compile(optimizer, loss, metrics=[metric])

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




# helper function to plot confusion matrix 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size=None):
    """ (Copied from sklearn website)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logging.info("Display normalized confusion matrix ...")
    else:
        logging.info('Display confusion matrix without normalization ...')


    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim([-0.5, len(classes)-0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm


    




def evaluate_model(model, classes, X_train, X_test, y_train, y_test):
    '''
    This function is to  Evaluate accuracy and time cost 
        Args : 
            model : model to evaluate on
            classes : dataset classes
            tr_X,tr_Y,te_X,te_Y : train's and test's (x and y)
    
    '''
    # start by recording time
    t0 = time.time()

    # accuracy on train set
    model.evaluate(X_train, y_train)

    # accuracy on test set
    model.evaluate(X_test, y_test)

    # get prediction as integers [0,1]
    y_test_predict  = np.array(model.predict(X_test)).astype(int)

    # git index of predicted values in each test sample
    y_test_predict = np.argmax(y_test_predict,axis=1)



    # git index of true values in each test sample
    y_test = np.array(y_test)
    y_test = np.argmax(y_test,axis =1)

    # Time cost
    average_time = (time.time() - t0) / (len(y_train) + len(y_test))
    logging.info("Time cost for predicting on train and test data is: "
          "{:.5f} seconds".format(average_time))

    # Plot confucion_matrix (TP,TN,FP,FN)
    plot_confusion_matrix(
        y_test, y_test_predict, classes, normalize=False, size=(12, 8))
    plt.show()





if __name__ == "__main__":

    
    with open('config.yaml') as f:
         config = yaml.load(f, Loader=SafeLoader)
    

    # get all needed configuration for training
    classes = config['classes']
    dataset_path = config['data_directory']
    test_size = config['test_size']
    sequence_length= config['sequence_length']
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss = config['loss']
    log_path = config['log_path']
    model_config= config['model']
    saved_weights_path=config['saved_weights_path']
    metric = config['metric']
    if metric == 'binary':
        metric = BinaryAccuracy()
    else :
        metric = Accuracy()


    # get training data from dataset folder
    X_train,X_test,y_train,y_test = getDataSet(classes,dataset_path,sequence_length,test_size)

    # get training model
    lstm = LSTM_model(model_config)

    # train model on our data
    model =Train(lstm,saved_weights_path,X_train,y_train,epochs,optimizer,loss,metric,log_path)

    # evaluate model on test data
    evaluate_model(model,classes,X_train,X_test,y_train,y_test)
    




