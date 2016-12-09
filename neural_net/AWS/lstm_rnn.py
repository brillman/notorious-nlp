import os.path, os, sys, numpy, json, random
#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from collections import Counter

#load the training corpus
def load_corpus(filename='lyrics.txt'):
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    return raw_text

#if you think there might be non-ascii or uppercase chars lurking in the data...
def clean_corpus(filename, output_name):
    """removes non-ascii characters from corpus file.  I assume that base-level pre-processing (e.g., removing
    punctuation to speed up training) has already occured"""
    raw_text = open(filename).read().lower()
    clean_text = ''.join([char if ord(char) < 128 else ' ' for char in raw_text])
    output_file = open(output_name, 'w')
    output_file.write(clean_text)
    print "File successfully written!"

#pre-process the data, step 1: create mappings from chars to int so that the model can understand a representation of the text
def create_char_dict(raw_text):
    """Process the text for training with a neural net.
    This function does this by mapping characters to integers"""
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    return {'chars':chars, 'char_to_int':char_to_int, 'int_to_char':int_to_char, 'n_chars':n_chars, 'n_vocab':n_vocab}

#pre-process the data, step 2: prepare the dataset that the model will train on 
def create_dataset(raw_text, char_to_int, n_chars, seq_length=100, patterns_per_file=1000000,
                   output_directory='dataset/patterns/', output_file_name='output_data_'):
    """The model predicts the upcoming characters from the seq_length (default: 100) characters that precedes it. This
    function formats the raw text file into patterns that the model can understand.
    
    Input values are seq_length characters that the precede the output value
    Output values single characters which follow a sequence of seq_length. The first output value is the seq_length+1-th
        character in the dataset
        
    parameters:
    - raw_text : string
    - char_to_int : dictionary of char to int mappings (e.g., 'a'->1)
    - n_char : int, number of chars mapped to int in char_to_int
    
    returns:
    - a dictionary including input:output pairs that can be fed to keratize_data()
    
    These patterns, when built from the full extent of my data, exceed 40BG.  To avoid holding/processing an object that
    large in memory this function pre-processes some segment of the data and writes it to disk.  Parameters for
    dividing the dataset into files are patterns_per_file, output_directory and output_file_name.  I found ~1M patterns
    (~750MB) to be managable.
    """
    dataX, dataY    =   [], []
    print "Finding patterns"
    for i in range(0, n_chars-seq_length, 1):
        seq_in = raw_text[i:i+seq_length]
        seq_out = raw_text[i+seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        #this block deals with writing files to disk. Cut/comment to streamline the process with smaller datasets
        if i%(patterns_per_file+1) == 0:
            print "{0} patterns found".format(i)
            dataset = {'X':dataX, 'Y':dataY}
            output_file = '{0}{1}_{2}.json'.format(output_directory, output_file_name, i/(patterns_per_file+1))
            with open(output_file, 'w') as file_:
                json.dump(dataset, file_)
            dataX, dataY = [], []
            print 'Patterns written to file; lists reset'
            #TO DO: modularize ln 62-68 so that we can cleanly write the remaining lines to disk
            #right now we lose them if patterns are loaded from files
            
    n_patterns = len(dataX)
    return {'X':dataX, 'Y':dataY, 'n_patterns':n_patterns, 'seq_length':seq_length}

#pre-process the data, step 3: transform your input:output patterns into what keras expects
def kerasize_data(dataset, n_vocab):
    """formats the data so the keras learning model can understand it.
    input_data: - reshaped to be a sequence of [samples, time steps, features] as expected by LSTM keras models
                - integer data normalized (range: 0-1) to speed up learning and better accomodate sigmoid activation functions
    output_data: - encoded as one-hot vectors of length n_vocab
    """
    #reshape the input data into vectors of [samples, time steps, features]
    X = numpy.reshape(dataset['X'], (dataset['n_patterns'], dataset['seq_length'], 1))
    #normalize reshaped data
    X = X / float(n_vocab)
    #encode output data as a one-hot vector
    y = np_utils.to_categorical(dataset['Y'])
    return {'X':X, 'y':y}

def define_model(X, y, drop_value=0.1, activation='softmax'):
    """Returns a Keras model.  Values I anticipate tweaking are input variables.  To change other aspects of
    the model (e.g., add another hidden layer), edit the guts of this function directly."""
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(drop_value))
    model.add(Dense(y.shape[1], activation=activation))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print "Model successfully defined"
    return model

def train_model(X, y, retrain_model=False, file_='weights/weights-improvement-t2-19.hdf5'):
    """trains a Keras model, storing the most promising weights after each epoch"""
    if not retrain_model:
        model = define_model(X, y)
    if retrain_model:
        model = load_model(file_, X, y)
        print "loading weights from: {0}".format(file_)
    filepath="weights/weights-improvement-t2-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=False, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(X, y, nb_epoch=30, batch_size=512, validation_split=0.01, callbacks=callbacks_list)

def load_model(filename, X, y):
    """Defines a new model (as specified in define_model) and then immediately assigns that model the
    best weights from a previous training session"""
    model = define_model(X, y)
    model.load_weights(filename)
    return model

def generate_hip_hop(model, raw_dataset, int_to_char, char_dict, random_seed=False):
    """Generates 500 characters of a hip hop song"""
    if random_seed:
        random.seed(0)
    #pick a random seed if you want results to be consistent across multiple runs
    start = numpy.random.randint(0, len(raw_dataset['X'])-1)
    pattern = raw_dataset['X'][start]
    print "Seed: \n", ''.join([int_to_char[value] for value in pattern]), '\n'

    #generate characters
    for i in range(500):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x/float(char_dict['n_vocab'])
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
 
#-------------------------------------------
#   FUNCTIONS TO DEAL WITH PATTERNS
#-------------------------------------------
def load_patterns(filelist, directory, seq_length=100):
    """my laptop doesn't have enough RAM to pre-process the text data into keras-compatible patterns.
    This function loads files with pre-processed data into memory"""
    dataX, dataY = [], []
    for filename in filelist:
        try:
            with open(directory+filename) as data_file:    
                values = json.load(data_file, encoding='utf8') 
            #print type(values)
            dataX.extend(values['X'])
            dataY.extend(values['Y'])
           # print len(dataX), len(dataY)
        except ValueError, e:       #catch non-json files
            print "something went wrong with filename {0}".format(filename)
    
    n_patterns = len(dataX)
    return {'X':dataX, 'Y':dataY, 'n_patterns':n_patterns, 'seq_length':seq_length}
     
def batch_train(char_dict, num_batches=8, directory='dataset/patterns/'):
    """This network is trained in multiple batches because of the size of the input data (80M characters). On an
    AWS GPU I can train ~100k char/min with keras batches of 512.
    
    This file trains the network in num_batches.  This file assumes that the entire raw dataset
    (of input/output vectors) has been pre-processed and stored in the specified directory. This
    allows the function to deal with datasets to hold in memory.
    
    To train the network in a single run, set num_batches=1"""
    filelist = os.listdir(directory)
    sub_lists = []
    
    #divide all the files in the directory into num_batches of about evenly sized lists
    for batch in range(num_batches):
        sub_list = filelist[batch::num_batches]
        sub_lists.append(sub_list)

    for batch, sub_list in enumerate(sub_lists):
        #load 1/num_batch of the pre-processed data into memory
        raw_dataset = load_patterns(sub_list, directory)
        #keratize that dataset
        dataset = kerasize_data(raw_dataset, char_dict['n_vocab'])
        X = dataset['X']
        y = dataset['y']
        int_to_char = char_dict['int_to_char']
        #train that model on that dataset
        if batch == 0:
            #build a new model the first time
            train_model(X, y, retrain_model=False)
        if batch > -1:
            #retrain all subsequent models
            train_model(X, y, retrain_model=True, file_='weights/weights-improvement-t2-19.hdf5')


def main(pre_process_data=False, generate_text=False, train_model=False, num_batches=2,
         lyrics_corpus='lyrics_corpus_clean.txt', model_weights='weights_12_8/weights-improvement-t2-29.hdf5'):
    """Pre-process raw text data into multiple GB of patterns, generate hip hop text or train a model in
    num_batches, depending on your preferences"""
    data_directory = 'dataset/patterns/'  
    with open('char_dict.json') as data_file:    
        char_dict = json.load(data_file) 
    
    #the dataset may be too large for a computer to load at once.  Pre-process the data and store
    #in multiple files to avoid this issue
    if pre_process_data:
        create_dataset(raw_text, char_dict['char_to_int'], char_dict['n_chars'])
    
    #train the model on the pre-processed data
    if train_model:
        batch_train(char_dict, num_batches=num_batches, directory=data_directory)
    
    #use the model to generate text
    if generate_text:
        raw_text = load_corpus(lyrics_corpus)
        char_dict = create_char_dict(raw_text) 
        #get a sample X, y dataset that the model can use as a seed for text generation
        #right now, the seed comes from a randomly pre_processed datafile in data_directory
        filelist = os.listdir(data_directory)
        #file_choice = random.choice(filelist)
        raw_dataset = load_patterns([random.choice(filelist)], data_directory)
        #kerasize dataset to generate X and y pairs
        dataset = kerasize_data(raw_dataset, char_dict['n_vocab'])
        X = dataset['X']
        y = dataset['y']
        int_to_char = char_dict['int_to_char']
        
        #define and load the weighted model you want to use for generation
        #just once
        model = load_model(model_weights, X, y) #weights-improvement-t2-19_old_v1.hdf5
        generate_hip_hop(model, raw_dataset, int_to_char, char_dict)
    
if __name__ == '__main__':
    #TO DO: Be a grown up and use sys_args here.  This is sloppy.
    #pre-process data and use it to train model
    #main(pre_process_data=True, train_model=True, num_batches=2)
    #train the model
    #main(train_model=True, num_batches=2)
    #generate text
    main(generate_text=True)