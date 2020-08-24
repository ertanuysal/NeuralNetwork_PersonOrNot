import os
import random
import numpy as np




# Reads the training data from the file in given path
# Input: path
# Output: list of words and their corresponding labels
path="./data"
def read_data(path):
	words = []
	labels = []
	with open(os.path.join(path, "train.txt"), "r") as input_file:
		for line in input_file:
			#check for empty lines
			if line.strip():
				words.append(line.strip().split("\t")[0])
    
				if line.strip().split("\t")[1] == "O":
					labels.append(0)
				else:
					labels.append(1)
	return words, labels




#Creates tuples from given words list in form: ( (n-1)th word, word, (n+1)th word, label ) for each sentence
#input: list of words and labels
#output: a list of shape (m, n, 4) where m is then number of sentences and n is the length of sentences
def create_samples(words, labels):
	started = False
	sentences = []
	sentence = []
	size = len(words)

	for i in range(len(words)):
		sample = []

		if words[i]==".":
			sentences.append(sentence)
			sentence = []
		else:
			#add (n-1)th word
			if i == 0 or words[i-1] == ".":
				sample.append("¡s¿")
			else:
				sample.append(words[i-1])

			#add middle word
			sample.append(words[i])

			#add (n+1)th word
			if i == size-1 or words[i+1] == ".":
				sample.append("¡/s¿")
			else:
				sample.append(words[i+1])

			#add label
			sample.append(labels[i])

			sentence.append(sample)

	sentences.append(sentence)#last sentence does not have a "."
	return sentences




#splits the data into test and training samples
#input: list of sentences of shape (m, n, 4) where m is then number of sentences and n is the length of sentences
#output: two numpy arrays of training and test samples, and two numpy arrays of labels 
def split_data(sentences, test_size=0.2):
	train_x =[]
	train_y = []
	test_x = []
	test_y = []

	#shuffle sentences
	random.shuffle(sentences)
	sep = int(len(sentences)*test_size)

	#concatenate each sample in the sentences together
	tmp_test = []
	for i in range(0,sep):
		tmp_test.extend(sentences[i])
	tmp_train = []
	for i in range(sep,len(sentences)):
		tmp_train.extend(sentences[i])

	#create numpy arrays
	train = np.array(tmp_train)
	train_x = train[:,0:3]
	train_y = train[:,3]
	test = np.array(tmp_test)
	test_x = test[:,0:3]
	test_y = test[:,3]

	return train_x, train_y, test_x, test_y
"""		
words, labels = read_data(path)
sentences = create_samples(words, labels)
train_x, train_y, test_x, test_y = split_data(sentences)

"""
















