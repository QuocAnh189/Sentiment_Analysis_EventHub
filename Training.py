import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

path = 'your_path_here.csv' #https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews

data_set = pd.read_csv(path)

y= data_set.iloc[:,0].values
X = data_set.iloc[:,1].values

from sklearn.model_selection import train_test_split
(X_train_text,X_test_text,y_train,y_test) = train_test_split(X,y, test_size = 0.3, random_state =  2020)

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(X)

X_train_token  = tokenizer.texts_to_sequences(X_train_text)
X_test_token = tokenizer.texts_to_sequences(X_test_text)

token_string = tokenizer.word_index
Inverse_map = dict(zip(token_string.values(), token_string.keys()))

def token_to_text(tokens):
    strings = [Inverse_map[token] for token in tokens if token != 0]
    text = " ".join(strings)
    return text

len_max = 241
#Padding the sequence 
X_train_pad = pad_sequences(X_train_token, maxlen = len_max, padding = 'pre')
X_test_pad = pad_sequences(X_test_token, maxlen = len_max, padding ='pre')

#Model
embedding_size = 20
Model = Sequential()
Model.add(Embedding(input_dim = 20000, input_length = len_max, output_dim = embedding_size, name = 'Embedding_Layer'))
Model.add(GRU(units = 16, return_sequences = True, name='First_Layer'))
Model.add(GRU(units = 8, return_sequences = True, name ='Second_layer'))
Model.add(GRU(units = 4, name = "Third_Layer"))
Model.add(Dense(1, activation = 'sigmoid', name ="Dense_Layer"))
optimizer = Adam(lr = 0.001)
Model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = optimizer)
Model.summary()

#Fitting the model
Model.fit(X_train_pad, y_train, validation_split = 0.05,epochs =5, batch_size = 32)
Model.summary()

result = Model.evaluate(X_train_pad,y_train)
Evaluate = Model.evaluate(X_test_pad,y_test)
print("The accuracy on the training set is {0:.2%}".format(result[1]))
print("The accuracy on the test set is {0:.2%}".format(Evaluate[1]))

def classifier(text):
    text_seq = tokenizer.texts_to_sequences(text)
    #text_seq = [np.array(a) for a in text_seq]
    text_token = pad_sequences(text_seq, maxlen = len_max, padding = 'pre')
    pred = Model.predict(text_token)
    if pred[0] >= 0.5:
        result = 'Positive Review! Thank you'
    else:
        result = "Negative Review! We will improve our delivery"
    return result

text = ["I love the movie"]
print(classifier(text))

#Saving the Model
Model.save("model/Model.h5")

import pickle 
with open('tokenize.pickle', 'wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)