import csv
import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import pythainlp
pythainlp.__version__

from pythainlp import sent_tokenize, word_tokenize

main_path = "[PATH_OF_SONG_FOLDER]"+ "\\"
allfiles = os.listdir("[PATH_OF_SONG_FOLDER]")+ "\\"
print(allfiles)


# preprocessing the corpus by converting all letters to lowercase, 
# replacing blank lines with blank string and removing special characters
def preprocessText(text):
    stopChars = [',','(',')','.','-','[',']','"','*']
    text = text.replace('\n','').replace('\t','').replace('\r','')
    processedText = text
    for char in stopChars:
        processedText = processedText.replace(char,' ')
    return processedText

#append Song to Dataframe
Total_song = pd.DataFrame()
for filename in allfiles:
    df = pd.read_csv(main_path +filename )
    Total_song = Total_song.append(df)

### Prepare data before train model ###
Text = '''()'.,!"/?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'''
for i in Text:
    # Total_song['lyrics'] = Total_song.lyrics.str.replace(' ' , '\n')
    Total_song['lyrics'] = Total_song.lyrics.str.replace(i , '')

#Change Column to list of lyrics
col_one_list = Total_song['lyrics'].tolist()

texts = []
for text in col_one_list:
    texts.append(preprocessText(text))  

Full_total_song = ''.join(texts)

#Token word
corpusList = word_tokenize(Full_total_song, keep_whitespace=False)
# print(corpusList)
map(str.strip, corpusList) #trim words
df_corpus = pd.DataFrame (corpusList, columns = ['corpus'])
df_corpus.to_csv('df_corpus.csv')
print('Done')

