import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
df=pd.read_csv('amazon.csv')

reviews=[]
labels=[]
for i in range(len(df)):
    reviews.append(df["Text"][i])  #append review to the list
    labels.append(df["label"][i])   #append label of a particular review to the list
labels=np.array(labels)             #convert  list to array
tokenizer=Tokenizer(num_words=200,oov_token="<OOV>")    #tokenising the words to numbers by word emebedding methodand adding OOV to the new words
tokenizer.fit_on_texts(reviews)                         #Fiiting the tokenizer
word_index = tokenizer.word_index                      
seq=tokenizer.texts_to_sequences(reviews)               #converting tokens to sequences  
pad_seq=pad_sequences(seq,padding='post',maxlen=10,truncating="post")   #adding extra tokens (usually zeros) to make all sequences in a dataset the same length

model = tf.keras.Sequential([
 tf.keras.layers.Input(shape=(10,)),  #accepting shape of review
 tf.keras.layers.Embedding(10000, 16),  #creates an embedding layer that maps 10,000 unique integer tokens to 16-dimensional dense vectors
 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), #Biderectional neural networks and LSTM
 tf.keras.layers.Flatten(), #Converting the array to a vector to be fed into fully connected layer
 tf.keras.layers.Dense(6, activation='relu'), 
 tf.keras.layers.Dense(1, activation='sigmoid') #activation function
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #loss is binary cross entropy as we are classifying binary classification
model.fit(pad_seq,labels,epochs=10)
# model.summary()
pred_review="It is bad"
tokenizer.fit_on_texts(pred_review)
pred_review=tokenizer.texts_to_sequences(pred_review)
pred_review=pad_sequences(pred_review,padding='post',maxlen=10,truncating="post")
avg=0
for i in range(len(model.predict(pred_review))):
    avg+=model.predict(pred_review)[i][0]

if avg/len(model.predict(pred_review))>0.5:
    print("Positive")
else:
    print("Negative")

