import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


class Tranform_New_Data:

    def __init__(self, features , artist_name="", song_title="" ):
        self.artist_name = self.tranform_str(artist_name)
        self.song_title = self.tranform_str(song_title)
        self.genius_url = f'https://genius.com/{self.artist_name}-{self.song_title}-lyrics'
        self.features = features

    def tranform_str(self, i_str):
        wthout_spc_chr = re.sub(r'[^A-Za-z0-9 ]+', '', i_str)
        return wthout_spc_chr.replace(' ','-', -1)
    
    def lyrics_scraping(self, url):

        request_genius = requests.get(url, timeout=1)

        if request_genius.status_code == 200:
            lyrics_soup = BeautifulSoup(request_genius.text, "html.parser")
        else:
            print("There is something wrong")

        genius_html = lyrics_soup.find_all('div', class_ ="Lyrics__Container-sc-1ynbvzw-1 kUgSbL")
        str_lyrics = ''.join(str(genius_html))

        pattern = r'>(.*?)<'

        matches = re.findall(pattern, str_lyrics)
        filtered_list = [item for item in matches if item != ""]
        return '\n'.join(filtered_list)
    
    def new_data(self):

        list_string =self.lyrics_scraping(self.genius_url)

        # Convert the content to lowercase (case-insensitive count)
        list_string = list_string.lower()

        # Split the content into words
        words = list_string.split()

        vocabulary_len = len(self.features)
        new_con = np.zeros(len(self.features)) 

        for index, word in enumerate(self.features):
            
            # Count the occurrences of the word
            new_con[index] = words.count(word.lower())

        new_df = pd.DataFrame([new_con], columns=self.features)
        new_df.drop('Explicit Content', axis=1, inplace=True)

        return new_df