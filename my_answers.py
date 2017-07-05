import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])   
        y.append(series[i + window_size]) 

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape= (window_size, 1) ))
    model.add(Dense(1, activation=None))
    # print(model.summary())
    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    unique_chars = list(set(text))
    # print(unique_chars)
    # remove as many non-english characters and character sequences as you can 
    # RUBRIC SAYS
    # The submission removes all non-english / non-punctuation characters. 
    # (English characters should include string.ascii_lowercase and punctuation includes 
    #  [' ', '!', ',', '.', ':', ';', '?'] 
    # (space, eclamation mark, comma, period, colon, semicolon, question mark))
    import re
    text = re.sub("[0123456789$%&*@'()/àâèé-]", ' ', text)
    text = text.replace('"',' ') #couldn't insert double quotes above because they were being used to wrap the regex
    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i : i + window_size])  
        outputs.append(text[i + window_size])
    

    
    return inputs,outputs
