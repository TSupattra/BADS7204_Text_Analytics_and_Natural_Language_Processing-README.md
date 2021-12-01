import bs4
import requests
import csv
import os
import pandas as pd

# Example link : 'https://xn--72c9bva0i.meemodel.com/%E0%B8%A8%E0%B8%B4%E0%B8%A5%E0%B8%9B%E0%B8%B4%E0%B8%99/%E0%B8%9B%E0%B9%89%E0%B8%B2%E0%B8%87%20%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%B4%E0%B8%99%E0%B8%97%E0%B8%A3%E0%B9%8C%20%E0%B8%81%E0%B8%B4%E0%B9%88%E0%B8%87%E0%B8%A8%E0%B8%B1%E0%B8%81%E0%B8%94%E0%B8%B4%E0%B9%8C'
link = "[Link of meemodel]"
data = requests.get(link)
soup = bs4.BeautifulSoup(data.text)
# print(soup)


all_song = soup.find('table',{'class':'table table-condensed table-striped'})
all_name = []
all_link = []
for row in all_song.find_all('tr'):
  try:
    name, artist, album, tube = row.find_all('td')
    all_name.append(name.text)
    all_link.append('https://xn--72c9bva0i.meemodel.com/' + name.find('a').get('href'))
  except:
    pass

# print(all_name)

all_lyrics = []
for link in all_link:
 data = requests.get(link)
 soup = bs4.BeautifulSoup(data.text)
 text = soup.find('div',{'class':'lyric'}).text
 all_lyrics.append(text)

song_list = pd.DataFrame(
    {'name': all_name,
     'link': all_link,
     'lyrics': all_lyrics
    })

song_list.to_csv(["Name_o_Artist.csv"])