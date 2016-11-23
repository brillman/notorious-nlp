#this module contains a number of helper functions called on by different modules
from artists import Artist, Song
import json
import os
import string


#LOADING AND BUILD OBJECTS
def load_file(filename):
    with open(filename) as data_file:    
        data = json.load(data_file, encoding='utf8') 
    return data

def make_text(directory='data/'):
    """
    transforms all the song lyrics present in a .json data file into a single string.
    this string can be fed into other modules, including word cloud creation functions
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

def make_corpus_from_directory(directory, output_name):
    """This function iterates over the json documents in data/ to build a .txt 'corpus' object
    that will be compatible with various nltk modules.
    
    this function produces a .txt file, so you should only have to call it once"""
    corpus_text = []
    
    for filename in os.listdir(directory):
        try:
            data = load_file(directory+filename)
            for song in data['songs']:
                if len(song["lyrics"]) > 10:    #some songs are missing lyrics; skip these
                    lyrics = replace_punctuation(song["lyrics"])
                    corpus_text.append(lyrics)
        
        except ValueError, e:       #catch non-json files
            print "something went wrong with filename {0}".format(filename)    
    
    output_file = open(output_name, 'w')        
    for text in corpus_text:
        output_file.write("%s\n" % text)
        
def build_artist(discography):
    """takes an artists discography and cleans each song represented.  Returns an Artist object
    associates with all songs in the discography (and whatever other information was orignally
    encoded in the discography)"""
    #discography = load_file(json_file)

    artist_name = discography['artist_name']
    all_songs = discography['songs']
    
    artist = Artist(artist_name)
    
    for song in all_songs:
        lyrics = replace_punctuation(song['lyrics'])    #ain't -> ain t
        if len(lyrics) > 10:
            song_title = song['song_title']
            s = Song(song_title, lyrics, artist_name)
            artist.add_song(s)
    
    return artist

def build_artists_list(directory='data/'):
    """calls build_artist() repeatedly to build a list of Artist objects representing each artist
    included in the corpus"""
    artist_list = []
    for filename in os.listdir(directory):
        try:
            discography = load_file(directory+filename)
            artist = build_artist(discography)
            artist_list.append(artist)
        
        except ValueError, e:       #catch non-json files
            print "something went wrong with filename {0}".format(filename)
    
    return artist_list
        
 #CLEANING       
def replace_punctuation(text, replace_with=' '):
    """Takes a string and replaces all instances of punctionation with the value specified in
    replace_with (string).
    
    The only two options that I've tested:
        replace_with = ' '  : ain't -> ain t
        replace_with = ''   : ain't -> aint
    """
    text = text.encode('utf-8')
    replace_function = string.maketrans(string.punctuation, replace_with*len(string.punctuation))
    text = text.translate(replace_function)
    return text.lower()

def count():
    i = ['neighba', 'make', 'pop', 'got', 'eye', 'true', 'real', 'hit', 'love', 'street', 'way', 'grindin',
         'will', 'kid', 'life', 'block', 'know', 'keep', 'hold', 'shit', 'stop', 'hard', 'man', 'time', 'rec',
         'good', 'girl', 'want', 'still', 'one', 'feel', 'cause', 'deck', 'respect', 'back', 'ya', 'take', 'll']
    m = ['yeah', 'love', 'right', 'll', 'go', 'now', 'neighba', 'back', 'game', 'way', 'new', 'got', 'yo', 'ya',
         ]