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
    reviews.append(df["Text"][i])
    labels.append(df["label"][i])
labels=np.array(labels)
tokenizer=Tokenizer(num_words=200,oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
seq=tokenizer.texts_to_sequences(reviews)
pad_seq=pad_sequences(seq,padding='post',maxlen=10,truncating="post")

model = tf.keras.Sequential([
 tf.keras.layers.Input(shape=(10,)),  
 tf.keras.layers.Embedding(10000, 16),
 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(6, activation='relu'),
 tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
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

