# coding:utf-8
import time
import numpy as np
import random
import json,pickle,jieba
import matplotlib
matplotlib.use('Agg')
from tqdm import *
import matplotlib.pyplot as plt 
#import word2vec
import pandas as pd

def get_stopwords(file_path):
	stopwords = []
	'''
	with open(file_path) as f:
		for word in f:
			word = word.strip("\n")
			stopwords.append(word)
	'''
	stopwords = ['了', '的']
	return stopwords


def remove_stopwords(stopwords,text):
	new_text = [ i for i in text if i not in stopwords]
	char_ret = []
	for word in new_text:
		for ch in word:
			char_ret.append(ch)
	#print('text:'+str(len(text)) +' '+str(len(new_text)))
	return char_ret


def load_text(text, max_sentence_len, input2idx, stopwords):
	text = list(jieba.cut(text))
	text = remove_stopwords(stopwords,text)
	text = [ i for i in text if i  in input2idx ]
        
	vectors = []
	for j in range(max_sentence_len):
		if j >= len(text):
			char = 'empty' # padding
		else:
			char = text[j]
		vectors.append(input2idx[char])
	vectors = np.asarray(vectors, dtype='int32')
	return vectors


def load_text_hierarchical(text, max_sentence_nb, maxlen, input2idx, choice):
	if choice == 'char':
		text = ''.join(filter(lambda x: x in input2idx, list(text)))
		text = text[4:]
	elif choice == 'word':
		text = list(jieba.cut(text))
		text = [ i for i in text if i  in input2idx ]
		
	# 分句
	seqs = text.split('。')
	hi_text = []
	for i in range(max_sentence_nb):
		if i >= len(seqs):
			seq_text = ['empty' for i in range(maxlen)]
		else:
			if len(seqs[i]) > maxlen:
				seq_text = seqs[i][:maxlen]
			else:
				seq_text = seqs[i]
		hi_text.append(seq_text)
	
	vectors = []
	for i in range(max_sentence_nb):
		vector = []
		for j in hi_text[i]:
			vector.append(input2idx[j])
		if len(vector) < maxlen:
			for k in range(maxlen-len(vector)):
				vector.append(input2idx['empty'])
		vectors.append(vector)
	vectors = np.asarray(vectors, dtype='int32')

	return vectors

def load_X(file_path, max_sentence_len, input2idx, stopwords):
	print('Loading x...')
	ret = []
	with open(file_path) as f:
		for i in tqdm(f):
			i = json.loads(i)['input_text']
			vectors = load_text(i, max_sentence_len, input2idx,stopwords)
			ret.append(vectors)
	ret = np.asarray(ret, dtype='int32')
	return ret


def load_X_hierarchical(file_path, max_sentence_nb, maxlen, input2idx, choice):
	print('Loading herarchical x...')
	ret = []
	with open(file_path) as f:
		for i in tqdm(f):
			i = json.loads(i)['input_text']
			vectors = load_text_hierarchical(i, max_sentence_nb, maxlen, input2idx,choice)
			ret.append(vectors)
	ret = np.asarray(ret, dtype='int32')
	return ret


def load_Y(file_path, mul_label_set_size, mul_label2idx):
	print("Loading mul y...")
	ret = [[] for i in range(4)]
	with open(file_path) as f:
		for i in tqdm(f):
			i = json.loads(i)['input_label']
			seg_i = [split_law(j) for j in i]
			for j in range(4):
				tem = []
				for split_res in seg_i:
					if j < len(split_res):
						tem.append(split_res[j])
				ret[j].append(n_hot_encoder(tem, mul_label_set_size[j], mul_label2idx[j]))

		for i in range(4):
			ret[i] = np.asarray(ret[i])
	return ret


def list2str(data):
	return '\n'.join(data)


def n_hot_encoder(data, set_size, char2idx):
	ret = np.zeros(set_size, dtype=np.int8)
	for i in data:
		idx = char2idx[i]
		ret[idx] = 1
	return ret


def n_hot_decoder(data, whole_set, top_nb=None, threshold=0.5, with_prob=False):
	ret = [(whole_set[i], data[i]) for i in range(len(data))]
	ret = sorted(ret, key=lambda x: x[1], reverse=True)
	filtered_ret = list(filter(lambda x: x[1]>threshold, ret))
	if with_prob == False:
		ret =list(map(lambda x: x[0], filtered_ret))
	else:
		ret = filtered_ret
	if top_nb is not None:
		ret = ret[:top_nb]
	return ret 


def plot_loss_figure(history, save_path):
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot(train_loss, 'b', val_loss, 'r')
	plt.xlabel('train_loss: blue   val_loss: red      epoch')
	plt.ylabel('loss')
	plt.title('loss figure')
	plt.savefig(save_path)



def get_input_set(file_path, min_count, stopwords):
	print("Getting input_set...")
	ret = []#set()
	with open(file_path) as f:
		for raw in tqdm(f):
			raw = json.loads(raw)['input_text']
			raw = list(jieba.cut(raw))
			for i in raw:
				ret.append(i)
	
	count = pd.Series(ret).value_counts()
	'''
	maxx = count.values[0]
	c = list(count)
	for i in range(1,maxx):
		print(str(i) + ' times: ' + str(c.count(i)))
	'''
	count = count[count >= min_count]
	ret = list(count.index)
	
	char_set = remove_stopwords(stopwords,ret)

	'''
	count = pd.Series(char_set).value_counts()
	count = count[count >= min_count]
	char_set = list(count.index)
	'''
	char_set = sorted(list(set(char_set)))
	input_set = ['empty']
	input_set.extend(list(char_set))
	input2idx = dict(zip(input_set, range(len(input_set))))
	return input_set, input2idx


def get_label_set(file_path):
	print("Getting label_set...")
	labels = [set() for i in range(4)]

	with open(file_path) as f:
		for raw in tqdm(f):
			raw = json.loads(raw)['input_label']
			for i in raw:
				seg_res = split_law(i)
				for j in range(len(seg_res)):
					labels[j].add(seg_res[j])
	labels = list(map(list, labels))
	for i in range(len(labels)):
		labels[i] = sorted(labels[i])
	mul_label2idx = [dict(zip(labels[i], range(len(labels[i])))) for i in range(4)]
	return labels, mul_label2idx


def split_law(law):
	sep_char = '条款项'
	ret = []
	idx = law.index('》')
	ret.append(law[:idx+1])
	pos = idx+1
	for i in range(pos, len(law)):
		if law[i] in sep_char:
			ret.append(law[:i+1])
			pos = i+1
	return ret

def unique(x):
	ret = []
	for i in range(len(x)):
		tag = True
		for j in range(i+1, len(x)):
			if x[j].startswith(x[i]):
				tag = False
				break
		if tag:
			ret.append(x[i])
	return ret

def concat(x):
	if type(x) is not list:
		return x
	data_set_size = len(x[0])
	ret = [[] for i in range(data_set_size)]
	for i in range(len(x)):
		for j in range(data_set_size):
			ret[j].append(x[i][j])
	return ret

def flatten(x):
	assert type(x) is list 
	ret = []
	for i in x:
		ret.extend(i)
	return ret

def decode(pred_res, label_set, top_nb, threshold, with_prob=False):
	if type(pred_res) is list:
		ret = [flatten([n_hot_decoder(j, label_set[idx], top_nb[idx], threshold[idx], with_prob) for idx,j in enumerate(i)]) for i in pred_res]
	else:
		ret = [n_hot_decoder(i, label_set, top_nb, threshold, with_prob) for i in pred_res]
	return ret


def load_matrix(bin_path, input2idx):
	model = word2vec.load(bin_path)
	vector_dim = model.vectors.shape[1]
	matrix = np.zeros((len(input2idx) , vector_dim))
	for word, i in input2idx.items():
    		embedding_vector = model[word]
    		if embedding_vector is not None:
        		matrix[i] = embedding_vector
	return matrix

def save_input2idx(dic,path):
	f = open(path,'wb')
	pickle.dump(dic, f)
	f.close()

def save_label2idx(dics,path):
	f = open(path, 'wb')
	n = len(dics)
	for i in range(n):
		pickle.dump(dics[i],f)
	f.close()

def load_input2idx(path):
	f = open(path, 'rb')
	input2idx = pickle.load(f)
	f.close()
	input_set = list(input2idx.keys())
	return input_set, input2idx

def load_label2idx(path):
	f = open(path,'rb')
	label2idx = []
	for i in range(4):
		label2idx.append(pickle.load(f))
	f.close()
	label_set = [list(label2idx[i].keys()) for i in range(4) ]
	return label_set, label2idx

def train_vector(input_set,vector_size, min_count, words_path, bin_path):
	f = open(words_path,'w')
	for i in input_set:
		f.write(i)
		f.write(' ')
	f.close()
	word2vec.word2vec(words_path, bin_path, size=vector_size,min_count=min_count, verbose=True)
	


def get_law_cons():
	xingfa = json.loads(open('../law_content/xingfa').readline())
	xinyongka = json.loads(open('../law_content/xinyongka').readline())
	laws = [xingfa, xinyongka]
	law_cons = {}	
	for i in laws:
		law_cons.update(i)
	return law_cons

def random_sample():
	ret = []
	with open('../train_data/credit_card_fraud_show') as f:
		for i in f:
			i = json.loads(i)['input_text']
			ret.append(i)
	nb = min(10000, len(ret))
	return random.sample(ret, nb)

