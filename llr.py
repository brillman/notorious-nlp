import os
import numpy as np
import scipy.stats as ss
from math import log
from collections import defaultdict, Counter
from word_cloud import make_word_cloud
#import matplotlib.pyplot as plt
import json
from artists import Artist, Song
from helper import build_artist, load_file

wu_tang_members = ['inspectah-deck', 'the-genius-gza', 'raekwon', 'method-man', 'ghostface-killah',
           'ol-dirty-bastard', 'cappadonna', 'wu-tang-clan']    #no u-god, I know

def build_wu_tang_clan(wu_tang_members):
    """builds Artist object for each member of the Wu-tang Clan. Returns a list of all Wu-tang Artist
    objects.
    I'm lazy, so this list is accessed through indices.  To explore the list, you can call wu_tang.get_name()
    for each wu-tang in the list."""
    wu_tang = []
    for member in wu_tang_members:
        address = 'data/'+member+'.json'
        discography = load_file(address)
        wu = build_artist(discography)
        wu.determine_word_frequencies()
        wu_tang.append(wu)
        
    return wu_tang

def build_comparison_sets(member, wu_tang):
    """returns word-count comparisons representations of two text objects
        one representing all the words used by member (member)
        one representing all the words used by other wu_tang members (wu_tang)"""
    member_dict = member.word_frequencies   #calls built-in method for the individual member
    #TO DO: figure out better way to handle this..
    #Right now: builds a Counter object that considers all word_frequencies for other members.
    #Inefficient, but it works
    wu_tang_dict = Counter()
    for wu in wu_tang:
        wu_tang_dict.update(wu.word_frequencies)
    return member_dict, defaultdict(int, wu_tang_dict)


def log_likelihood_ratio(member, wu_tang):
    """
    Takes two dictionaries of word frequency (one from the member, one from the rest of the clan)
    to produce a dictionary of words and their LLR weights.  The math actually yeilds the *inverse*
    of LLR, because of how I choose to visualize the data.  To the best of my knowledge, the core
    relationships that a non-inverted LLR would get at are preserved here.
    
    (inverted) LLR formula:
        L_w = N_w*log(N_w/E_w) / N_bar*log(N_bar/E_bar)
        -> change / to + for 'canonical' LLR
    L_w = LLR score for word w
    n_i = number of words in the input member's discography
    k_i = number of times word w was used in the input members discography
    n_b = number of words in wu-tang clan's background discography
    n_b = number of times word w was used in the wu-tang clan's background discography
    """
    member_len = sum(member.values())
    wu_tang_len = sum(wu_tang.values())
    LLRs = {}
    for word in member.keys():
        #print word
        #member probability
        n_i = float(member_len)
        k_i = float(member[word])
        try:
            p_i = k_i/n_i
        except ZeroDivisionError, e:
            p_i = 0
        prob_i = ss.binom.pmf(n=n_i, k=k_i, p=p_i)
        #print n_i, k_i, p_i, prob_i
        
        #wu-tang probability
        n_b = float(wu_tang_len)
        k_b = float(wu_tang[word])
        try:
            p_b = k_b/n_b
        except ZeroDivisionError, e:
            p_b = 0
        prob_b = ss.binom.pmf(n=n_b, k=k_b, p=p_b)
        
        LLR = k_i*log(prob_i)/k_b*log(prob_b)
        LLRs[word]=LLR
        #print word, LLR
    
    return LLRs

def make_LLR_word_cloud(llr_json, image_path='wu-tang.jpg', bg_color='black', max_words=75):
    """Uses the LLR values of words in an artist's discography to produce an LLR-weighted
    word cloud for that artist.  Does this by creating a new text representation of that
    artist's discography, with word frequency weighted by LLR and then building a word cloud
    from that representation
    
    Other parameters are direct imports from make_word_cloud.  See that function's documentation
    for more details"""
    #create new text representation
    txt = ''
    LLR = load_file(llr_json)
    for word in LLR.keys():
        if LLR[word] > 0:
            string = [word for i in range(int(LLR[word]*10))]
            txt+=' '.join(string)
    
    #use that text to make a word cloud
    make_word_cloud(txt, image_path=image_path, bg_color=bg_color, max_words=max_words)

def main():
    """Builds an LLR-weighted word cloud for each artist in the Wu-tang Clan.  In addition, also creates
    a json file that stores the inverted LLR values for each artist."""
    wu_tang = build_wu_tang_clan(wu_tang_members)
    for i in range(len(wu_tang)):
        wu = wu_tang[i]
        tang = wu_tang[:]
        del tang[i]
        wu_dict, wu_tang_dict = build_comparison_sets(wu, wu_tang)
        LLRs = log_likelihood_ratio(wu_dict, wu_tang_dict)
        output_name = wu_tang_members[i]+'_llr.json'
        with open(output_name, 'w') as file_:
            json.dump(LLRs, file_, indent=4)
        make_LLR_word_cloud(output_name)

if __name__ == '__main__':
    main()
    