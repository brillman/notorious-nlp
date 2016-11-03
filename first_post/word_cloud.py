#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from artists import Artist, Song
from measure_vocab import build_artist
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

def make_word_cloud(text, image_path="drake_head.png", background_color="white", max_words=2000,
                    mask=coloring, max_font_size=40):
    """
    Generates the actual word cloud
        text = text to be modeled (str)
        image_path = path to image whose silhouette the generator will use
        background_color = don't fill in space with this color
        max_words = int (how many words in the cloud)
        max_font_size = int (how big does text get)
    """
    # read the mask / color image
    coloring = np.array(Image.open(path.join(d, "ready_to_die.jpg")))
    stopwords = set(STOPWORDS)
    
    wc = WordCloud(background_color="white", max_words=2000, mask=coloring,
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

def main():
    text = make_text()
    make_word_cloud()
    
if __name__ == '__main__':
    main()
