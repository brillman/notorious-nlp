from artists import Artist, Song
import json
import string
import os

#corpus stats: ~31k songs; ~300 artists, ~6.25m words (estimate)

def load_file(filename):
    with open(filename) as data_file:    
        data = json.load(data_file)
        
    #print type(data)
    return data

def build_artist(json_file):
    """takes a .json representation of an artist's discography and builds an Artist object with it
    assumes the .json has at least a value for 'artist_name' (string) and a value for 'songs'
    (list of dictionaries with at least entries for 'song_title' and 'lyrics'"""
    discography = load_file(json_file)

    artist_name = discography['artist_name']
    songs = discography['songs']
    
    artist = Artist(artist_name)
    
    for song in songs:
        lyrics = clean_lyrics(song['lyrics'])
        if len(lyrics) > 10:
            song_title = song['song_title']
            s = Song(song_title, lyrics, artist_name)
            artist.add_song(s)
    
    return artist
    
def clean_lyrics(lyrics):
    """takes song lyrics (string) and encodes them properly.  Also strips out capitals
    and punctuation"""
    lyrics = lyrics.encode('utf-8')
    lyrics = (lyrics.translate(string.maketrans("",""), string.punctuation)).lower()
    return lyrics


def build_artists_list(directory='data/'):
    """Iterates through every json data file to build and return a list of all artists whose
    discographies are represented.  It counts the numbers of artists, songs, and words in the
    corpus along the way
    
    Artists will be sorted alphabetically in the output list"""
    num_artists, num_songs, num_words = 0, 0, 0
    artists = []

    #iterate through every file in the data directory
    for filename in os.listdir('data/')[1:]:   
        try:
            artist = build_artist('data/'+filename)
            num_artists += 1
            if num_artists % 100 == 0:      
                print "We're still working! Currently counting {0}'s discography".format(artist.get_name())
            num_songs += len(artist.songs)
            num_words += artist.total_words()
           # if artist.average_unique_words() != 0:
           #     artists.append(artist)
        #my scraper resulted in a few empty .json files (if the artists had no lyrics/songs associated with them)
        #we want to ignore those files
        except AttributeError, e:  
            pass
        except ValueError, e:       #.DS_STORE, .git files, etc throw an error
            print "something went wrong with filename {0}".format(filename)
          
    print "Corpus contains:\n\t{0} artists\n\t{1} songs\n\t {2} words ".format(num_artists, num_songs, num_words)

    return artists


#The following are nearly identical functions for ranking artists by different artists methods.
#It might be worth building an artist_collection objects that can store these (and similar functions)
#as methods.  I'll make a note if I got back and change that in artists.py.  If so, this code will be
#redundant

def rank_by_unique_words(artists, display=10):
    """Takes a list of Artist objects, ranks them by average number of unique words per song.
    Returns the re-ordered list of artists
    
    Display determines how many results are printed"""
    artists = sorted(artists, key=lambda artist:artist.average_unique_words())
    
    #comment out print statements if you don't care
    for artist in artists[:display]:
        print "{0}'s unique words/song: {1} unique words/song (on average)".format(artist.get_name(), artist.average_unique_words())
          
    for artist in artists[-display:]:
        print "{0}'s unique words/song: {1} unique words/song (on average)".format(artist.get_name(), artist.average_unique_words())
    
    return artists

def rank_by_average_song_length(artists, display=10):
    """Takes a list of artist objects, ranks them by average song length.  Returns the re-ordered list.
    
    Display determines how many artists are printed out"""
    artists = sorted(artists, key=lambda artist:artist.average_song_length())
    
    for artist in artists[:display]:
        print "{0}'s avg song length: {1} unique words/song (on average)".format(artist.get_name(), artist.average_song_length())
     
    for artist in artists[-display:]:
        print "{0}'s avg song length: {1} unique words/song (on average)".format(artist.get_name(), artist.average_song_length())
    
    return artists

def rank_by_vocab_ratio(artists, display=10):
    """Takes a list of artist objects and ranks them by a ratio of unique words:total words, computed across
    every song where data is available.  """
    artists = sorted(artists, key=lambda artist:artist.average_vocab_ratio())
    
    for artist in artists[:display]:
        print "{0}'s vocab ratio: {1} unique words/song (on average)".format(artist.get_name(), artist.average_vocab_ratio())
          
    for artist in artists:
        print "{0}'s vocab ratio: {1} unique words/song (on average)".format(artist.get_name(), artist.average_vocab_ratio())
        
    return artists
    
def rank_by_most_songs(artists, display=10):
    artists = sorted(artists, key=lambda artist:len(artist.songs))
    
    for artist in artists[:display]:
        print "{0} has {1} songs in the corpus".format(artist.get_name(), len(artist.songs))
          
    for artist in artists:
        print "{0} has {1} songs in the corpus".format(artist.get_name(), len(artist.songs))
        
    return artists

def main():
    artists = build_artists_list()
    artists = rank_by_unique_words(artists)
    artists = rank_by_average_song_length(artists)
    artists = rank_by_vocab_ratio(artists)
    artists = rank_by_most_songs(artists)
    
if __name__ == '__main__':
    main()
    