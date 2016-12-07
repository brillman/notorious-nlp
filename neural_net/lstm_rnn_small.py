import os.path, sys, numpy, json, random
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from helper import load_file

from collections import Counter

def clean_corpus(filename, output_name):
    """removes non-ascii characters from corpus file"""
    raw_text = open(filename).read().lower()
    clean_text = ''.join([char if ord(char) < 128 else ' ' for char in raw_text])
    output_file = open(output_name, 'w')
    output_file.write(clean_text)
    print "File successfully written!"

#load the training corpus
def load_corpus(filename='nyt_articles.txt'):
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    return raw_text

#process the data
def create_char_dict(raw_text):
    """Process the text for training with a neural net.
    This function does this by mapping characters to integers"""
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    return {'chars':chars, 'char_to_int':char_to_int, 'int_to_char':int_to_char, 'n_chars':n_chars, 'n_vocab':n_vocab}

#prepare the dataset that the model will train on by transforming the raw text to
#input and output pairs all encoded as intergers 
def create_dataset(raw_text, char_to_int, n_chars, seq_length=200):
    """The model predicts the upcoming characters from the seq_length (default: 100) characters that precedes it.
    This function runs through the dataset and splits the input and output values from the raw text.
    Input values are seq_length characters that the precede the output value
    Output values single characters which follow a sequence of seq_length. The first output value is the seq_length+1-th
        character in the dataset"""
    dataX, dataY    =   [], []
    print "Finding patterns"
    for i in range(0, n_chars-seq_length, 1):
        seq_in = raw_text[i:i+seq_length]
        seq_out = raw_text[i+seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        #this section was added to deal with memory limits on my laptop
        if i%1000000 == 0:
            print 'Writing current data to file; reseting list'
            output_name = 'neural_net/dataset/data_output_{0}.txt'.format(i/1000000)
            output_file = open(output_name, 'w')
            for j in range(len(dataX)):
                x_datum = dataX[j]
                y_datum = dataY[j]
                output_file.write("{0}\t{01}\n".format(x_datum, y_datum))
            dataX, dataY = [], []
            
            print i, "patterns found"
            
    n_patterns = len(dataX)
    return {'X':dataX, 'Y':dataY, 'n_patterns':n_patterns, 'seq_length':seq_length}

def kerasize_data(dataset, n_vocab):
    """This function formats the data so the keras learning model can understand it.
    input_data: - reshaped to be a sequence of [samples, time steps, features] as expected by LSTM keras models
                - integer data normalized (range: 0-1) to speed up learning, and better accomodate sigmoid activation functions
    output_data: - encoded as one-hot vectors of length n_vocab"""
    #reshape the input data as [samples, time steps, features]
    X = numpy.reshape(dataset['X'], (dataset['n_patterns'], dataset['seq_length'], 1))
    #normalize reshaped data
    X = X / float(n_vocab)
    #encode output data as a one-hot vector
    y = np_utils.to_categorical(dataset['Y'])
    return {'X':X, 'y':y}

def define_model(X, y, drop_value=0.2, activation='softmax'):
    """Returns a Keras model.  Values I anticipate tweaking are input variables.  To change other aspects of
    the model, edit the guts of this function directly."""
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(drop_value))
    model.add(Dense(y.shape[1], activation=activation))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print "Model successfully defined"
    return model

def train_model(X, y, retrain_model=False):
    """trains a Keras model, storing the best weights each epoch"""
    if not retrain_model:
        model = define_model(X, y)
    if retrain_model:
        model = load_model('neural_net/weights/weights-improvement-14.hdf5', X, y)
        print "loading weights from: 'neural_net/weights/weights-improvement-14.hdf5'"
    filepath="neural_net/weights/weights-improvement-trial2-{epoch:02d}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint = ModelCheckpoint(filepath, verbose=0, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(X, y, nb_epoch=20, batch_size=128, validation_split=0.01, callbacks=callbacks_list)

def load_model(filename, X, y):
    """Defines a new model and then immediately assigns that model that best weights from a previous training"""
    model = define_model(X, y)
    #filename = "wonderland_weights/weights-improvement-19-1.8595.hdf5"     #this will be a result of the training
    model.load_weights(filename)
    return model

def generate_hip_hop(model, raw_dataset, int_to_char, char_dict, random_seed=False):
    """Generates 1000 characters of a hip hop song"""
    if random_seed:
        random.seed(0)
    #pick a random seed if you want results to be consistent across multiple runs
    start = numpy.random.randint(0, len(raw_dataset['X'])-1)
    pattern = raw_dataset['X'][start]
    print "Seed: \n", ''.join([int_to_char[value] for value in pattern]), '\n'

    #generate characters
    for i in range(1000):
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
def load_patterns(filename):
    dataX, dataY = [], []
    sizes = Counter()
    loaded = 0
    for line in open(filename, 'r'):
        line = line.split('\t')
        if len(line) >  1:
            dataX.append(list(line[0]))
            dataY.append(line[-1])
            sizes[len(list(line[0]))] += 1
            loaded += 1
        if loaded % 100000 == 0:
            print loaded, "lists loaded"
    n_patterns = len(dataX)
    print sizes
    seq_length = 200
    return {'X':dataX, 'Y':dataY, 'n_patterns':n_patterns, 'seq_length':seq_length}
     
#-------------------------------------------
#   FUNCTIONS TO DEAL WITH PATTERNS
#-------------------------------------------
    
def main(load_pickle=False, generate_text=False, retrain_model=False):
    #clean_corpus('txt_corpus/lyrics_corpus.txt', 'txt_corpus/lyrics_corpus_clean.txt')
    #load text
    raw_text = load_corpus('txt_corpus/lyrics_corpus_clean.txt')
    char_dict = create_char_dict(raw_text)
    
    if not load_pickle:
        #translate text into a dataset
        #Uncomment the next five line to test create_char_dict and see the vocabulary the model will be trained on
        print "Total characters: ", char_dict['n_chars']
        print "Total vocab: ", char_dict['n_vocab']
        print "Vocab:"
        for v in char_dict['chars']:
            print '\t', unicode(v, "utf-8")
        raw_dataset = create_dataset(raw_text, char_dict['char_to_int'], char_dict['n_chars'])
        print "Total patterns: ", raw_dataset['n_patterns']
        #save the patterns so that you don't need to run this step multiple times
        pickled_patterns = 'neural_net/pickled_patterns.json'
        with open(pickled_patterns, 'w') as outfile:
            json.dump(raw_dataset, outfile, indent=4)
    
    if load_pickle:
        #raw_dataset = load_file("neural_net/pickled_patterns.json")
        raw_dataset = load_patterns('neural_net/dataset/patterns/patterns.txt')
        print "Loaded patterns"

    #kerasize dataset to generate X and y pairs
    dataset = kerasize_data(raw_dataset, char_dict['n_vocab'])
    X = dataset['X']
    #print dataset['X'][15]
    y = dataset['y']
    #print dataset['y'][15]
    int_to_char = char_dict['int_to_char']
    
    if not generate_text:
        train_model(X, y, retrain_model=retrain_model)
        
    if generate_text:    
        model = load_model('neural_net/weights/weights-improvement-13.hdf5', X, y)
        generate_hip_hop(model, raw_dataset, int_to_char, char_dict)
    
if __name__ == '__main__':
    #generate text
    #main(load_pickle=True, generate_text=True)
    #train new model using pickled patterns
    main(load_pickle=True)
    #train new model
    #main(load_pickle=False)
    #re-train a previous model
    #main(load_pickle=True, retrain_model=True)