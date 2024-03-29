''' 
This file is for creating model and train it using tensorFlow 

    Functions :
        LSTM_model(modelConfig)
        getDataSet(classes,datasetPath,sequence_length,test_size)
        Train(model,model_path,X_train,y_train,epochs,optimizer,loss,metric,logsPath)

'''

import yaml
import time
import pickle
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense,Dropout
from tensorflow.python.keras.metrics import Precision


if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    from utils.plot_lib import plot_confusion_matrix,plt_statistic






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
        elif item['layer'] == 'Dense' :
            model.add(Dense(item['units'],activation=item['activation']))
        else:
            model.add(Dropout(item['drop_perc']))
        

        

    return model

def getDataSet(classes,datasetPath,sequence_length,test_size,test=False):
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
    y = np.array(labels).astype(np.float32)
   

    if not test:
        # split DataSet to train test 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=15)

        return X_train,X_test,y_train,y_test
    else:
         return X,y
   




# tensorflow callback to cancel training if validation_loss <=0.02
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_loss')<=0.02):
      print("\nloss is less thatn 0.02 so cancelling training!")
      self.model.stop_training = True



def Train(model,model_path,X_train,y_train,X_val,y_val,epochs,optimizer,loss,metric):
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
        
        Return : trained model 

    '''


    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    callback =myCallback()
    metrics = [Precision(),metric]
    
    # if weights are already caclulated then load them to the model and return it 
    if os.path.exists(model_path) == True:

        # load weights into model
        model.load_weights(model_path)

        #compile model
        model.compile(optimizer, loss, metrics=metrics)

        return model

    


    # compile the model
    model.compile(optimizer, loss, metrics=metrics)

    # start training
    history =model.fit(X_train, y_train, epochs= epochs,batch_size=16 ,callbacks=[tb_callback,callback]   ,validation_data = (X_val,y_val))

    # save model weights after training
    model.save("models/weights.h5")

    # save model history after saving it
    with open('models/history.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    


    return model





    




def evaluate_model(model,history, classes, X_train, X_val,X_test, y_train, y_val,y_test):
    '''
    This function is to  Evaluate accuracy and time cost 
        Args : 
            model : model to evaluate on
            classes : dataset classes
            tr_X,tr_Y,te_X,te_Y : train's and test's (x and y)
    
    
    '''

    fig = plt.figure()


    t0 = time.time()
    
    # accuracy on train set
    model.evaluate(X_train, y_train)

    # accuracy on test set
    model.evaluate(X_val, y_val)

    # get prediction as integers [0,1]
    y_val_predict  = np.array(model.predict(X_val)).astype(int)

    y_test_predict  = np.array(model.predict(X_test)).astype(int)


    # git index of predicted values in each test sample
    # y_val_predict = np.argmax(y_val_predict,axis=1)



    # git index of true values in each test sample
    # y_val = np.array(y_val)
    # y_val = np.argmax(y_val,axis =1)

    # Time cost
    average_time = (time.time() - t0) / (len(y_train) + len(y_val))
    logging.info("Time cost for predicting on train and test data is: "
        "{:.5f} seconds".format(average_time))

    
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

        # Plot confucion_matrix (TP,TN,FP,FN)
    plot_confusion_matrix(ax2,
        y_val, y_val_predict, classes, cmap =plt.cm.Purples ,normalize=False,title="validation set confusion matrix")
        # Plot confucion_matrix (TP,TN,FP,FN)
    plot_confusion_matrix(ax1,
        y_test, y_test_predict, classes,normalize=False,title="test set confusion matrix")
    
    # plot accuracy
    plt_statistic(history,ax4,'loss',True)

    # plot loss
    plt_statistic(history,ax3,'accuracy',True)

  
    plt.show()




def main(config):
     # get all needed configuration for training
    classes = config['classes']
    dataset_path = config['data_directory']
    test_size = config['test_size']
    sequence_length= config['sequence_length']
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss = config['loss']
    model_config= config['model']
    saved_weights_path=config['saved_weights_path']
    test_path = config['test_set_path']
    metric = 'accuracy'
    if os.path.exists('models/history.history'):
      history = pickle.load(open('models/history.history','rb'))
    model = None
    
    # get training data from dataset folder
    X_train,X_val,y_train,y_val = getDataSet(classes,dataset_path,sequence_length,test_size)
    if test_path != "":
      X_test,y_test = getDataSet(classes,test_path,sequence_length,test_size,True)
    

    lstm = LSTM_model(model_config)

    print(lstm.summary())
    # train model on our data
    model =Train(lstm,saved_weights_path,X_train,y_train,X_val,y_val,epochs,optimizer,loss,metric)

        # evaluate model on test data
    evaluate_model(model,history,classes,X_train,X_val,X_test,y_train,y_val,y_test)

    



if __name__ == "__main__":

    
    with open('config.yaml') as f:
         config = yaml.load(f, Loader=SafeLoader)

    main(config=config)
   



