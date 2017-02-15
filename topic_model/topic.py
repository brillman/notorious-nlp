#from helper import make_text

import sys, re, collections
from os import listdir
from os.path import isfile, join
import numpy as np
from operator import itemgetter

### Global structures to keep track of words and their mappings'''
word2Index = {}
vocabulary = []
vocab_size = 0
NUM_TOPICS = int(sys.argv[2])
NUM_DOCS = 0 

#including a manual stopwords list because nltk is heavy
stopWords_English = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves',
			 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
			 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
			 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
			 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
			 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
			 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
			 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
			 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
			 'now']

#including a list of hip hop specific stopwords, brough to us by the LLR script
stopWords_hiphop = ['m', 'll', 'go', 'you', 'see', 'like', 'put', 'let', 'get', 'chorus', 'ain', 'ya',
					'hit', 'want', 'wanna', 'one', 'cause', 'way', 're', 'us', 'know', 'got']

stopWords = stopWords_English+stopWords_hiphop

def load_file(filename):
	"""loads a(n already tokenized) file and adjusts global variables accordingly.  Returns a list of all tokens
	in the file"""
	global vocab_size, vocabulary
	words = file(filename).read().lower().split()
	words = [w for w in words if w.isalpha() and w not in stopWords] #remove non-alpha and stopwords

	tokens = []
	for w in words:
		if w not in word2Index:			
			word2Index[w] = vocab_size
			vocabulary.append(w)
			vocab_size += 1
		tokens.append(word2Index[w])

	return tokens

def load_directory(directory):
	"""loads all files in a given directory.  Use this to find topics across all songs by all artists"""
	global NUM_DOCS
	file_data = [] # list of file data

	file_list = [ f for f in listdir(directory) if isfile(join(directory,f)) ]

	for f in file_list:
		file_data.append(load_file(join(directory,f)))
		NUM_DOCS += 1

	return file_data

def load_artists_directory(artist):
	"""loads all songs assocatied with a given artist.  Use this to find topics across all songs by a single artist""" 
	pass


def e_step(file_data, theta_t_z, theta_z_w):
	"""computes the expected value of the variables, given the current model parameters"""
	count_t_z = np.zeros([NUM_DOCS, NUM_TOPICS])
	count_w_z = np.zeros([vocab_size, NUM_TOPICS])

	for t in range(NUM_DOCS):
		"""to improve efficiency the scripts goes through each document only once to calculate the posterior distributions
		as and when necessary. The variable 'posterior_w_z' below is implicitly representing P(z | w, t) for the current
		document t."""
		posterior_w_z = collections.defaultdict(lambda:np.zeros(NUM_TOPICS))

		for w in file_data[t]:
			if w not in posterior_w_z:
				#calculate the posterior probability P(z | w, t)
				for z in range(NUM_TOPICS):
					posterior_w_z[w][z] = theta_t_z[t][z] * theta_z_w[z][w]

				#normalize the posterior
				posterior_w_z[w] /= np.sum(posterior_w_z[w])

			for z in range(NUM_TOPICS):
				#Update soft counts n(t, z)
				count_t_z[t][z] += posterior_w_z[w][z]

				#Update soft counts n(w, z)
				count_w_z[w][z] += posterior_w_z[w][z]

	return count_t_z, count_w_z


def m_step(count_t_z, count_w_z):
	"""	"""
	# Max Likelihood estimate of theta_t_z
	theta_t_z = np.copy(count_t_z)
	for t in range(NUM_DOCS):
		theta_t_z[t] /= np.sum(theta_t_z[t])

	# Max Likelihood estimate of theta_z_w
	theta_z_w = np.transpose(count_w_z)
	for z in range(NUM_TOPICS):
		theta_z_w[z] /= np.sum(theta_z_w[z])

	return theta_t_z, theta_z_w




def EM(file_data, num_iter):

	#Initialize parameters
	theta_t_z = np.random.rand(NUM_DOCS, NUM_TOPICS)
	theta_z_w = np.random.rand(NUM_TOPICS, vocab_size)

	#normalize
	for t in range(NUM_DOCS):
		theta_t_z[t] /= np.sum(theta_t_z[t])
	for z in range(NUM_TOPICS):
		theta_z_w[z] /= np.sum(theta_z_w[z])


	for i in range(num_iter):
		print "Iteration", i+1, '...'
		count_t_z, count_w_z = e_step(file_data, theta_t_z, theta_z_w)
		theta_t_z, theta_z_w = m_step(count_t_z, count_w_z)

	return theta_t_z, theta_z_w



if __name__ == '__main__':
	input_directory = sys.argv[1]
	file_data = load_directory(input_directory)
	num_iter = int(sys.argv[3])

	print "Vocabulary:", vocab_size, "words."
	print "Running EM with", NUM_TOPICS, "topics."
	theta_t_z, theta_z_w = EM(file_data, num_iter)

	#Print out topic samples
	for z in range(NUM_TOPICS):
		
		wordProb = [(vocabulary[w], theta_z_w[z][w]) for w in range(vocab_size)]
		wordProb = sorted(wordProb, key = itemgetter(1), reverse=True)

		print "Topic", z+1
		for j in range(20):
			print wordProb[j][0], '(%.4f),' % wordProb[j][1], 
		print '\n'
