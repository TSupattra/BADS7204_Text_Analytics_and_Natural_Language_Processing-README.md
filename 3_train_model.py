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


# The maximum length sentence we want for a single input in words
seqLength = 8
examples_per_epoch = len(corpusList)//(seqLength + 1) # number of seqLength+1 sequences in the corpus

# Create training / targets batches
wordDataset = tf.data.Dataset.from_tensor_slices(word_as_int)
print(wordDataset)
sequencesOfWords = wordDataset.batch(seqLength + 1, drop_remainder=True) # generating batches of 8 words each, typically converting list of words (sequence) to string
print(sequencesOfWords)

def split_input_target(chunk): # This is where right shift happens
  input_text = chunk[:-1]
  print(input_text)
  target_text = chunk[1:]
  print(target_text)
  return input_text, target_text # returns training and target sequence for each batch

dataset = sequencesOfWords.map(split_input_target) # dataset now contains a training and a target sequence for each 8 word slice of the corpus


#define BATCH_SIZE and BUFFER_SIZE
BATCH_SIZE = 64 # each batch contains 64 sequences. Each sequence contains 8 words (seqLength)
BUFFER_SIZE = 100 # Number of batches that will be processed concurrently
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in words
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of GRU units
rnn_units = 1024

def createModel(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = createModel(vocab_size = len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}") 

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#Train Model
EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = createModel(len(vocab), embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()



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
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(' ' + idx2words[predicted_id])

  return (startString + ''.join(text_generated))


#save trained model for future use (so we do not have to train it every time we want to generate text)
model.save('model_generate_song.h5')




