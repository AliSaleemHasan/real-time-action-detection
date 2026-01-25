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
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,LSTM, Dense,Dropout
from tensorflow.keras.metrics import Precision

import sys
import os
from utils.plot_lib import plot_confusion_matrix,plt_statistic
try:
    from utils.gpu_helper import configure_gpu
    from src.config_schema import Config
    # Configure GPU immediately to ensure memory growth is set before TF initializes
    configure_gpu()
except ImportError:
    print("Warning: Could not import gpu_helper. Running without explicit GPU configuration.")







logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


from src.models.factory import ModelFactory

def create_model_from_config(config, sequence_length=30):
    """
    Creates the model using the factory based on configuration.
    """
    # Create a wrapper or modify factory to accept the full config or just sub-part
    # The factory expects the full config dictionary to check 'architecture'
    # But relies on 'model' key for the layers.
    # We'll pass the whole config object that we get in main()
    
    # We need to construct a config object that matches what ModelFactory expects
    # In main(), 'config' is passed.
    pass # This function is a placeholder if we needed one, but better to just use ModelFactory directly in main.

# LSTM_model function removed as it is now in src/models/lstm.py


def getDataSet(classes,datasetPath,sequence_length,test_size,is_multiclass=False,test=False):
    '''
    This function is used to get dataset labels and sequences and split it to 
    input X and output(labels) Y

        Args:   
            classes : our dataset classes
            datasetPath: path for our saved dataset
            sequence_length: number of frame in each video
            test_size: size for testSet
            is_multiclass: if True, applies one-hot encoding
        
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

        # Capture all folders in the action path
        folder_path = os.path.join(datasetPath,action)
        if not os.path.exists(folder_path):
             print(f"--- [Training Data] Warning: Action folder {action} not found. Skipping. ---")
             continue
             
        all_items = os.listdir(folder_path)
        # Filter: Only keep folders that are numeric (e.g. "0", "1", "2")
        video_folders = [f for f in all_items if f.isdigit()]
        
        if not video_folders:
            print(f"--- [Training Data] Warning: No numeric feature folders found in {action}. Skipping. ---")
            continue

        # get all video for action by looping over all folders for this action 
        for videoFolder in np.array(video_folders).astype(int):
            
            video_path = os.path.join(datasetPath, action, str(videoFolder))
            
            # Robustness: Check if folder contains enough files
            if len(os.listdir(video_path)) < sequence_length:
                print(f"--- [Training Data] Warning: Video {videoFolder} in {action} has fewer frames than sequence length ({sequence_length}). Skipping. ---")
                continue

            # make window list to save all frame files for one video  
            window = []
            is_video_valid = True

            # loop  number of frames times
            for frame_num in range(sequence_length):
                npy_file_path = os.path.join(video_path, "{}.npy".format(frame_num))
                
                # Robustness: Check file existence and size
                if not os.path.exists(npy_file_path):
                    print(f"Warning: Missing frame {frame_num} in {video_path}. Skipping video.")
                    is_video_valid = False
                    break
                
                if os.path.getsize(npy_file_path) == 0:
                    print(f"Warning: Empty file {frame_num} in {video_path}. Skipping video.")
                    is_video_valid = False
                    break

                try:
                    # get frame features that stored as np_array
                    res = np.load(npy_file_path)
                    window.append(res)
                except Exception as e:
                    print(f"Error loading {npy_file_path}: {e}. Skipping video.")
                    is_video_valid = False
                    break

            if is_video_valid:
                # add all frame features for one video of one action to sequences list
                sequences.append(window)

                # add label for this video to labels
                labels.append(labelMap[action])

    # create X as np array of sequences
    X = np.array(sequences)

    # y is labels 
    y = np.array(labels).astype(np.float32)
    
    # Apply One-Hot Encoding if multiclass (Softmax)
    if is_multiclass:
        y = to_categorical(y, num_classes=len(classes))
   

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



def Train(model,model_path,X_train,y_train,X_val,y_val,epochs,batch_size,optimizer,loss,metric):
    '''
    This function is to train predefiend model and save training statistic 

        Args:
            model : predefiend model to train 
            model_path: path for previous saved weights if found
            X_train : data to train on 
            y_train : labels of data to train on 
            epoches : number of epochs to train the model
            batch_size : batch size for training
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
    history =model.fit(X_train, y_train, epochs= epochs,batch_size=batch_size ,callbacks=[tb_callback,callback]   ,validation_data = (X_val,y_val))

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

    # get prediction as probabilities
    y_val_probs = model.predict(X_val)
    y_test_probs = model.predict(X_test)


    # Determine if Multiclass or Binary based on output shape
    # If final dimension > 1, it's multiclass (Softmax). If == 1, it's binary (Sigmoid).
    if y_val_probs.shape[-1] > 1:
        y_val_predict = np.argmax(y_val_probs, axis=1)
        y_test_predict = np.argmax(y_test_probs, axis=1)
    else:
        # Binary Classification logic
        y_val_predict = (y_val_probs > 0.5).astype(int).flatten()
        y_test_predict = (y_test_probs > 0.5).astype(int).flatten()


    # Handle Ground Truth
    # If using categorical_crossentropy, y_val is One-Hot (2D). 
    # If using sparse_categorical or binary, it might be 1D or (N,1).
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        y_val = np.argmax(y_val, axis=1)
    else:
        y_val = np.array(y_val).flatten().astype(int)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    else:
        y_test = np.array(y_test).flatten().astype(int)

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




def main(config: Config):
     # get all needed configuration for training
    classes = config.classes
    dataset_path = config.data_directory
    test_size = config.test_size
    sequence_length= config.sequence_length
    epochs = config.epochs
    optimizer = config.optimizer
    loss = config.loss
    batch_size = config.batch_size
    # config.model is list of objects, we need to convert to list of dicts for legacy compat if needed
    # or just use it if factory supports it. For now, matching detect.py approach:
    model_config = [layer.model_dump() for layer in config.model]
    saved_weights_path=config.saved_weights_path
    test_path = config.test_set_path
    metric = 'accuracy'
    if os.path.exists('models/history.history'):
      history = pickle.load(open('models/history.history','rb'))
    model = None
    
    # get training data from dataset folder
    is_multiclass = model_config[-1]['activation'] == 'softmax'
    X_train,X_val,y_train,y_val = getDataSet(classes,dataset_path,sequence_length,test_size,is_multiclass=is_multiclass)
    if test_path != "":
      X_test,y_test = getDataSet(classes,test_path,sequence_length,test_size,is_multiclass=is_multiclass,test=True)
    

    # Create model using factory
    model_factory = ModelFactory()
    # We need to inject the input shape into the first layer config if it's not handled by the Interface
    # The Interface .create_model took input_shape.
    # But our Factory.get_model returns a wrapper class, not the Keras model itself yet.
    # Correction: Factory.get_model returns the Wrapper instance (LSTMModel or CNN1DModel).
    # We need to call .create_model() on it.
    
    action_model_wrapper = ModelFactory.get_model(config.model_dump())
    lstm = action_model_wrapper.create_model(input_shape=(sequence_length, 51))

    print(lstm.summary())
    # train model on our data
    model =Train(lstm,saved_weights_path,X_train,y_train,X_val,y_val,epochs,batch_size,optimizer,loss,metric)

    if test_path != "":
        # evaluate model on test data
      evaluate_model(model,history,classes,X_train,X_val,X_test,y_train,y_val,y_test)

    


if __name__ == "__main__":

    
    with open('config.yaml') as f:
         config_data = yaml.load(f, Loader=SafeLoader)
         config = Config(**config_data)

    main(config=config)
   



