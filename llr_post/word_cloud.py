#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from artists import Artist, Song
from helper import build_artist, load_file
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

d = os.path.dirname(__file__)

def make_text(directory='data/'):
    """
    transforms all the song lyrics present in the .json data into a single string.
    this string can be fed to the word_cloud module
    """
    text = ''
    for filename in os.listdir(directory):
        try:
            artist = build_artist(directory+filename)
            for song in artist.songs:
                text += song.get_lyrics()
        except AttributeError, e:
            pass
        except ValueError, e:
            print "something went wrong with filename {0}".format(filename)
    
    return text

def make_word_cloud(text, image_path="drake_head.png", bg_color="white", max_words=2000,
                    max_font_size=40):
    """
    Generates the actual word cloud
        text = text to be modeled (str)
        image_path = path to image whose silhouette the generator will use
        background_color = don't fill in space with this color
        max_words = int (how many words in the cloud)
        max_font_size = int (how big does text get)
    """
    # read the mask / color image
    coloring = np.array(Image.open(os.path.join(d, image_path)))
    stopwords = set(STOPWORDS)
    
    wc = WordCloud(background_color=bg_color, max_words=2000, mask=coloring,
                   stopwords=stopwords, max_font_size=40, random_state=42)
    # generate word cloud
    wc.generate(text)
    
    # create coloring from image
    image_colors = ImageColorGenerator(coloring)
    
    # show
    plt.imshow(wc)
    plt.axis("off")
    plt.figure()
    # recolor wordcloud and show
    # we could also give color_func=image_colors directly in the constructor
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure()
    plt.imshow(coloring, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
    
def make_wu_tang_clouds():
    #I know, no U-God.  I'm sorry :(
    members = ['inspectah-deck', 'the-genius-gza', 'raekwon', 'method-man', 'ghostface-killah', 'ol-dirty-bastard', 'rza', 'masta-killa', 'cappadonna', 'wu-tang-clan']
    wu_tang = []
    for member in members:
        address = 'data/'+member+'.json'
        discography = load_file(address)
        wu = build_artist(discography)
        wu_tang.append(wu)
       
    for member in wu_tang:
        text = member.get_lyrics_as_text()
        make_word_cloud(text, image_path='wu-tang.jpg', bg_color='black')
        

def main():
    #text = make_text()
    #make_word_cloud()
    make_wu_tang_clouds()
    
if __name__ == '__main__':
    main()
