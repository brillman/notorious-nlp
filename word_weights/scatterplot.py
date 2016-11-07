import numpy as np
import matplotlib.pyplot as plt
import os
from helper import load_file, build_artist, build_artist_list
from word_weights import get_weights, build_artist, load_file


weights = get_weights()
artists = build_artists_list()

N = len(artists)

#Hand-picked artists to label
#TO DO: make a more interactive version of this chart using pyxley or d3.js
label_these = ['destinys-child','the-dream','desiigner','lep-bogus-boys','yg','mablib','black-moon',
'bobby-shmurda','cunninlynguists','sean-price','earl-sweatshirt','raekwon','ghostface-killah',
'black-milk','mike-g','open-mike-eagle','madvillain','roc-marciano','aesop-rock','the-genius-gza',
'special-ed','krs-one','the-roots','pitbull','the-weeknd','rihanna','ciara','lauryn-hill','usher',
'salt-n-pepa','wu-tang-clan','nerd','twista','too-short','solange-knowles','watsky','xzibit',
'yukmouth','tyler-the-creator','run-dmc','wiz-khalifa','dmx','sugarhill-gang','cee-lo','young-thug',
'ty-dolla-sign']

#figure out which datapoints in the plot to label
indexes = []
for index, artist in enumerate(artists):
    if artist.get_name() in label_these:
        indexes.append(index)

#assign each dot in the plot a random unique color
colors = [np.random.rand(1,3) for i in range(N)]

data = [(artist.get_hip_hop_score(weights), artist.average_vocab_ratio()) for artist in artists]
data = np.array(data)

plt.xlabel('Hip hop score')
plt.ylabel('Vocabulary size (ratio)')
plt.title('Hip hop scores by vocab size')

labels = [artist.get_name() for artist in artists]
plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    data[:, 0], data[:, 1], marker = 'o',
    c = [color[0] for color in colors], s = .1*750,
    alpha = .25,
    cmap = plt.get_cmap('Spectral'))

meta_data = zip(labels, data[:,0], data[:,1], colors)

#for label, x, y, color in zip(labels, data[:, 0], data[:, 1], colors):
for index in indexes:
    label, x, y, color = meta_data[index]
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = color[0], alpha = 0.5),
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        )

plt.show()