# -*- coding: utf-8 -*-
"""InVelocità.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rP7eGIVlwJZOW9C_RSAMj6fDQO1ARjEo
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
import spacy
import matplotlib
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from pprint import pprint
from collections import Counter
import string
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
import spacy
import matplotlib
 
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, LSTM
 
from keras.preprocessing.sequence import pad_sequences
 
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
 
from sklearn.model_selection import KFold
 
 
# %matplotlib inline

!wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wr2DC3jRIwtmd-_kt-kbskDNgFHL0RBS' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=154yVk9HaQI1hxZorjZlJUQtGpPba5Cfn" -O emotion.csv && rm -rf /tmp/cookies.txt

df = pd.read_csv("emotion.csv")
df.head(10)

len(df)

len(df)

value_count = df['emotions'].value_counts()
value_count

value_count.plot(kind='bar', title = 'Emotions Count')

uno = df.loc[df['emotions'] == 'joy']
due = df.loc[df['emotions'] == 'sadness']
tre = df.loc[df['emotions'] == 'anger']
quattro = df.loc[df['emotions'] == 'fear']
cinque = df.loc[df['emotions'] == 'love']
sei = df.loc[df['emotions'] == 'surprise']

df2=pd.concat([uno[1:14972],due[1:14972],tre[1:14972],quattro[1:14972],cinque[1:14972],sei[1:14972]])

arr = np.arange(len(df2))
out = np.random.permutation(arr) # random shuffle

df2=df2.iloc[out]
df3=df
df=df2

"""### Convert from dataframe to list"""

len(df)

sentence_list = [t for t in df['text'].to_list()]
tag_list = [e for e in df['emotions'].to_list()]

"""#### The input sentences."""

sentence_list[:2]

tag_list[:2]

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

new_sentence=[spacy_tokenizer(sentence) for sentence in sentence_list]



cv=CountVectorizer()
word_count_vector=cv.fit_transform(sentence_list)

word_count_vector.shape

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights'])

df_idf.to_csv('df_idf.csv')

words = []
for sentence in sentence_list:
    for w in sentence.split():
        words.append(w)
    
words = list(set(words))
print(f"Size of word-vocablury: {len(words)}\n")

words

value_count = df['emotions'].value_counts()
value_count

value_count.plot(kind='bar', title = 'Emotions Count')

word2idx=dict(dict(df_idf)['idf_weights'])

word2idx

tags = []
for tag in tag_list:
    tags.append(tag)
tags = list(set(tags))
print(f"Size of tag-vocab: {len(tags)}\n")
print(tags)

tag2idx = {word: i for i, word in enumerate(tags)}
print(tag2idx)

sentence_list = [' '.join(y for y in x if len(y) > 1 ) for x in new_sentence]

sentence_list[:10]

X = [[word2idx[w] for w in s.split(' ') if w in word2idx.keys()] for s in sentence_list]

X

len(word2idx)

len(set(list(word2idx.values())))

word2idx

y = [tag2idx[t] for t in tag_list]
y[:3]



features = np.zeros((len(X), 30), dtype='float64')

for i, row in enumerate(X):
    if len(row) > 0:
        features[i][-len(row):] = row[:30]

features

#X_train, y_train = features[0:int(len(X)*0.7)], y[0:int(len(X)*0.7)]
X_test, y_test = features[int(len(X)*0.7+1):], y[int(len(X)*0.7+1):]
X_train, y_train = features, y



from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train.shape, len(y_train)

num_words=len(word2idx)

maxlen=30

X_train = pad_sequences(X_train, maxlen = maxlen)
#X_test = pad_sequences(X_test, maxlen = maxlen)

acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=10, shuffle=True)
fold_no = 1
for train, test in kfold.split(X_train, y_train):
  model = Sequential()
  model.add(Embedding(num_words, 20, input_length=30))
  model.add(Conv1D(filters=512, kernel_size=3, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=3, strides=1))
  model.add(Dropout(0.5))
  model.add(Conv1D(filters=256, kernel_size=4, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=4, strides=1))
  model.add(Dropout(0.5))
  model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=5, strides=1))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(6, activation='sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', 'binary_accuracy']) #optimizer adam se solo cnn o rmsprop se con lstm
  earlystopping= EarlyStopping(min_delta=0.001, patience=10)
  history = model.fit(X_train[train], y_train[train], batch_size=64,  epochs=50, callbacks=[earlystopping], validation_split=0.2)
  scores = model.evaluate(X_train[test], y_train[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

from sklearn.metrics import confusion_matrix

y_pred = model.predict_classes(X_train[test], 64, verbose=0)

y_test_bis = np.argmax(y_train[test], axis=1) 
y_test_bis

conf = confusion_matrix(y_test_bis, y_pred)

conf

model.save('modelkeras.model')

!zip -r /content/model.zip /content/modelkeras.model/

print(1050/1528)
print(506/1460)
print(55/1490)
print(341/1554)
print(824/1453)
print(366/1497)

(0.69+0.35+0.04+0.22+0.57+0.24) /6

import seaborn as sns
sns.heatmap(conf, annot=True)

count_y_test_label=Counter(y_test_bis)

93/1541

264/1506

293/1463

conf

from sklearn.metrics import classification_report

y_pred = model.predict_classes(X_train[test], 64, verbose=0)

y_pred_test_bool = np.argmax(y_train[test], axis=1)
print(classification_report(y_pred_test_bool, y_pred_bool))

y_train[test]

y_pred

model.save('/content/model.h5')

