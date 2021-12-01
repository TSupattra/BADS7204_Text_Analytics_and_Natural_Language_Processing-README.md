import csv
import os
import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
import numpy as np
import os
import time

#road file corpus
df_corpus = pd.read_csv('[PATH_OF_FILE_df_cospus.csv]' )

#df_corpus to list
corpusList = df_corpus['corpus'].tolist()


vocab = sorted(set(corpusList))
print('Corpus length (in words):', len(corpusList))
print('Unique words in corpus: {}'.format(len(vocab)))
word2idx = {u: i for i, u in enumerate(vocab)}
idx2words = np.array(vocab)
word_as_int = np.array([word2idx[c] for c in corpusList])



def generateLyrics(model, startString, temp,num_generate):
  print("---- Generating lyrics starting with '" + startString + "' ----")
  # Number of words to generate
  num_generate = num_generate

  # Converting our start string to numbers (vectorizing)
  start_string_list =  [w for w in startString.split(' ')]
  input_eval = [word2idx[s] for s in start_string_list]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # temp represent how 'conservative' the predictions are. 
      # Lower temp leads to more predictable (or correct) lyrics
      predictions = predictions / temp 
      # print(predictions)
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # print(predicted_id)

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      # print(input_eval)

      text_generated.append(' ' + idx2words[predicted_id])
      

  return (startString + ''.join(text_generated))

# load Model
from keras.models import load_model
model_w = load_model('[PATH_OF_MODEL_model_generate_song.h5]')
#save trained model for future use (so we do not have to train it every time we want to generate text)

print(generateLyrics(model_w, startString="คิดถึง", temp=0.6,num_generate = 100))

while (True):
  print('Enter start string:')
  input_str = input().lower().strip()
  temp=0.6
  num_generate = 100
  print(generateLyrics(model_w, startString=input_str, temp=temp,num_generate=num_generate))
  







