from collections import defaultdict
import json

class Song(object):
    """representation of a song"""
    def __init__(self, title, lyrics, artist, album='Album unknown'):
        self.title = title
        self.lyrics = lyrics
        self.artist = artist
        self.album = album
        self.num_words = len(lyrics.split())
        
    def get_title(self):
        return self.title
    
    def get_lyrics(self):
        return self.lyrics
    
    def get_artist(self):
        return self.artist
    
    def get_num_unique_words(self):
        unique_words = defaultdict(int)
        for lyric in self.lyrics.split(' '):
            unique_words[lyric] += 1
            
        return len(unique_words.keys())
    
    def get_hip_hop_score_song(self, word_weights):
        """returns the hip hop score for a given song by averaging the hip hop
        weights of all words in that song."""
        num_words, total_score = 0, 1
        for word in self.get_lyrics():
            try:
                score = word_weights[word]['score']
                total_score += score
                num_words += 1
            except KeyError, e:
                pass
        
        final_score = total_score/float(num_words)
        return final_score
    
    
class Album(object):
    def __init(self, name, artist, songs=[], label='Label unknown', release_year=1776):
        self.name = name
        self.artist = artist
        self.songs = songs
        self.label = label
        self.release_year = release_year
        
    
class Artist(object):
    def __init__(self, name, ascii_name=False, label='Label unknown'):
        self.name = name
        if ascii_name is False:
            self.ascii_name = name
        else:
            self.ascii_name = ascii_name
        self.label = 'Label unknown'
        self.songs = []
        self.albums = []
    
    #Get functions    
    def get_name(self):
        return self.name
    
    def get_ascii_name(self):
        return self.ascii_name
    
    def get_songs(self):
        return self.songs
    
    def set_ascii_name(self, new_ascii_name):
        self.ascii_name = new_ascii_name
        print "{0} ascii name set to {1}".format(self.get_name(), self.get_ascii_name())
    
    def add_song(self, song):
        """adds a song (Song) to an artist's repertoire"""
        self.songs.append(song)
    
    def total_words(self):
        total_words = 0
        for song in self.songs:
            total_words += song.num_words
        return total_words
        
    def average_song_length(self):
        num_words = 0
        for song in self.songs:
            num_words += song.num_words
            
        avg_num_words = num_words/len(self.songs)
        return avg_num_words
    
    def average_unique_words(self):     
        num_words = 0
        for song in self.songs:
            num_words += song.get_num_unique_words()
        
        try:
            avg_num_unique_words = num_words/len(self.songs)
        except ZeroDivisionError, e:
            avg_num_unique_words = 0
        return avg_num_unique_words
    
    def average_vocab_ratio(self):
        try:
            ratio = self.average_unique_words()/float(self.average_song_length())
        except ZeroDivisionError, e:
            ratio = 0
        
        return ratio
    
    def get_hip_hop_score(self, word_weights):
        """returns a score of how 'hip hop' an artist's vocabulary is, by measuing the hip hop
        score of each individual word they use"""
        num_songs = len(self.songs)
        total_score = 0
        for song in self.songs:
            score = song.get_hip_hop_score_song(word_weights)
            total_score += score
    
        try:
            final_score = total_score/float(num_songs)
        except ZeroDivisionError, e:
            final_score = 0
        
        return final_score
    
