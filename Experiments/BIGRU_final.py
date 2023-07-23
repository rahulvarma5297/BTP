# %%
import pandas as pd
import numpy as np
from keras import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, Masking, GRU
import numpy as np
import keras
from keras.utils import np_utils
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
import datetime
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import load
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


np.random.seed(1337)# setting the random seed value

# %%
# Mounting Drive
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# path_dataset = "drive/My Drive/DL_project_LJP/ILDC_multi.csv" # path to dataset

# %%
dataset = pd.read_csv('ILDC_multi.csv') # loading datasetc

# %%
# path to transformer generated chunk embeddings eg. XLNet etc.
path_transformer_chunk_embeddings_train = 'XLNet_full/XLNet_train.npy' 
path_transformer_chunk_embeddings_dev = 'XLNet_full/XLNet_dev.npy'
path_transformer_chunk_embeddings_test = 'XLNet_full/XLNet_test.npy'

# %%
# loading the chunk embeddings
x_train0 = load(path_transformer_chunk_embeddings_train, allow_pickle = True)
x_dev0 = load(path_transformer_chunk_embeddings_dev, allow_pickle= True)
x_test0 = load(path_transformer_chunk_embeddings_test, allow_pickle= True)

# %%
# loading the corresponding label for each case in dataset
dev = dataset.loc[dataset['split'] == 'dev'] 
train = dataset.loc[dataset['split'] == 'train'] 
test = dataset.loc[dataset['split'] == 'test'] 

y_train0 = []
for i in range(train.shape[0]):
    y_train0.append(train.loc[i,'label'])  
    
y_dev0 = []
for i in range(dev.shape[0]):
    y_dev0.append(dev.loc[i+32305,'label'])

y_test0 = []
for i in range(test.shape[0]):
    y_test0.append(test.loc[i+33299,'label'])

# %%
from keras.layers import Input, GRU, Bidirectional, Dense, Masking, MaxPooling2D, Reshape
from keras import layers


text_input = Input(shape=(None,768,), dtype='float32', name='text')
l_mask = Masking(mask_value=-99.)(text_input)

# After masking we encoded the vector using 2 bidirectional GRU's
encoded_text = layers.Bidirectional(GRU(100,return_sequences=True))(l_mask)
encoded_text = layers.Bidirectional(GRU(100,return_sequences=True))(encoded_text)
# Add a Conv2D layer to increase height and width of tensor
encoded_text = layers.Conv2D(filters=200, kernel_size=(3,3), padding='same')(encoded_text)
# Add MaxPooling2D layer to downsample the tensor
encoded_text = layers.MaxPooling2D(pool_size=(2, 2))(encoded_text)
encoded_text1 = layers.Bidirectional(GRU(100,))(encoded_text)
# Added a dense layer after encoding
out_dense = layers.Dense(30, activation='relu')(encoded_text1)
# And we add a sigmoid classifier on top
out = layers.Dense(1, activation='sigmoid')(out_dense)
# At model instantiation, we specify the input and the output:
model = Model(text_input, out)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()






# %%
num_sequences = len(x_train0)
batch_size = 32 
batches_per_epoch =  int(num_sequences/batch_size)
num_features= 768
def train_generator(): # function to generate batches of corresponding batch size
    x_list= x_train0
    y_list =  y_train0
    # Generate batches
    while True:
        for b in range(batches_per_epoch):
            longest_index = (b + 1) * batch_size - 1
            timesteps = len(max(x_train0[:(b + 1) * batch_size][-batch_size:], key=len))
            x_train = np.full((batch_size, timesteps, num_features), -99.)
            y_train = np.zeros((batch_size,  1))
            # padding the vectors with respect to the maximum sequence of each batch and not the whole training data
            for i in range(batch_size):
                li = b * batch_size + i
                x_train[i, 0:len(x_list[li]), :] = x_list[li]
                y_train[i] = y_list[li]
            yield x_train, y_train

# %%
num_sequences_val = len(x_dev0)
batch_size_val = 32
batches_per_epoch_val = int(num_sequences_val/batch_size_val)
num_features= 768
def val_generator():# Similar function to generate validation batches
    x_list= x_dev0
    y_list =  y_dev0
    # Generate batches
    while True:
        for b in range(batches_per_epoch_val):
            longest_index = (b + 1) * batch_size_val - 1
            timesteps = len(max(x_dev0[:(b + 1) * batch_size_val][-batch_size_val:], key=len))
            x_train = np.full((batch_size_val, timesteps, num_features), 0)
            y_train = np.zeros((batch_size_val,  1))
            # padding the vectors with respect to the maximum sequence of each batch and not the whole validation data
            for i in range(batch_size_val):
                li = b * batch_size_val + i
                x_train[i, 0:len(x_list[li]), :] = x_list[li]
                y_train[i] = y_list[li]
            yield x_train, y_train

# %%
# Setting the callback and training the model
call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=2, verbose=2,
                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)

model.fit_generator(train_generator(), steps_per_epoch=batches_per_epoch, epochs=3,
                    validation_data=val_generator(), validation_steps=batches_per_epoch_val, callbacks =[call_reduce] )

# %%
num_sequences_test = len(x_test0)
batch_size_test = 32
batches_per_epoch_test = int(num_sequences_test/batch_size_test) + 1
num_features= 768

def test_generator(): # function to generate batches of corresponding batch size
    x_list= x_test0
    y_list =  y_test0
    # Generate batches
    while True:
        for b in range(batches_per_epoch_test):
            if(b == batches_per_epoch_test-1): # An extra if else statement just to manage the last batch as it's size might not be equal to batch size 
              longest_index = num_sequences_test - 1
              timesteps = len(max(x_test0[:longest_index + 1][-batch_size_test:], key=len))
              x_train = np.full((longest_index - b*batch_size_test, timesteps, num_features), -99.)
              y_train = np.zeros((longest_index - b*batch_size_test,  1))
              for i in range(longest_index - b*batch_size_test):
                  li = b * batch_size_test + i
                  x_train[i, 0:len(x_list[li]), :] = x_list[li]
                  y_train[i] = y_list[li]
            else:
                longest_index = (b + 1) * batch_size_test - 1
                timesteps = len(max(x_test0[:(b + 1) * batch_size_test][-batch_size_test:], key=len))
                x_train = np.full((batch_size_test, timesteps, num_features), -99.)
                y_train = np.zeros((batch_size_test,  1))
                # padding the vectors with respect to the maximum sequence of each batch and not the whole test data
                for i in range(batch_size_test):
                    li = b * batch_size_test + i
                    x_train[i, 0:len(x_list[li]), :] = x_list[li]
                    y_train[i] = y_list[li]
            yield x_train, y_train

# %%
# evaluating on the test data
test_loss, test_acc=  model.evaluate_generator(test_generator(), steps= batches_per_epoch_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)


# %%
# defining a function which calculates various metrics such as micro and macro precision, accuracy and f1
def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    TP = []
    FP = []
    FN = []
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[i][j]

        FN.append(summ)
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[j][i]

        FP.append(summ)
    for i in range(0,2):
        TP.append(cm[i][i])
    precision = []
    recall = []
    for i in range(0,2):
        precision.append(TP[i]/(TP[i] + FP[i]))
        recall.append(TP[i]/(TP[i] + FN[i]))

    macro_precision = sum(precision)/2
    macro_recall = sum(recall)/2
    micro_precision = sum(TP)/(sum(TP) + sum(FP))
    micro_recall = sum(TP)/(sum(TP) + sum(FN))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
    print("macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1")
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

# %%
# getting the predicted labels on the test data
preds = model.predict_generator(test_generator(), steps= batches_per_epoch_test)
y_pred = preds > 0.5

# Calculating all metrics on test data predicted label
print(metrics_calculator(y_pred, y_test0[:-1]))

# %%
# getting the predicted labels on the dev data
preds = model.predict_generator(val_generator(), steps= batches_per_epoch_val)
y_pred_dev = preds > 0.5

# Calculating all metrics on dev data predicted label
print(metrics_calculator(y_pred_dev, y_dev0[:-2]))

# %%
# saving the trained model
model.save('BIGRU_XLNet.h5')  # creates a HDF5 file 'BIGRU_XLNet.h5'

# %%
# loading the model
# model = load_model('BIGRU_XLNet.h5')


