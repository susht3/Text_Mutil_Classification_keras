import sys
sys.path.append("../model/")
import os
import time
import json, pickle
import numpy as np
from datetime import datetime
from itertools import product
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from util import plot_loss_figure, load_X_bag_of_words, load_Y, n_hot_decoder, unique
from util import get_input_set, list2str, get_label_set, flatten, concat
from util import save_input2idx, save_label2idx, load_input2idx, load_label2idx,train_vector,load_matrix
from lr import build_logistic
from mlp import build_mlp

# @profile
def decode(pred_res, label_set, top_nb, threshold, with_prob=False):
	if type(pred_res) is list:
		ret = [flatten([n_hot_decoder(j, label_set[idx], top_nb[idx], threshold[idx], with_prob) for idx,j in enumerate(i)]) for i in pred_res]
	else:
		ret = [n_hot_decoder(i, label_set, top_nb, threshold, with_prob) for i in pred_res]
	return ret

# @profile
def test(model, test_X, test_y, label_set):
	data_set_size = len(test_X)
	pred_res = model.predict(test_X)
	if type(test_y) is list:
		top_nbs = [[2, 10, 10, 6], [2, 10, 10, 10]]
		thresholds = [[0.2,0.2,0.2,0.2], [0.3,0.3,0.3,0.3],[0.4, 0.4, 0.4, 0.4]]
		# top_nbs = product(range(6,11), repeat=4)
		# thresholds = [[0.2,0.2,0.2,0.2],[0.4,0.4,0.4,0.3]]
		test_y = concat(test_y)
		test_y = [flatten([n_hot_decoder(j, label_set[idx]) for idx,j in enumerate(i)]) for i in test_y]
		pred_res = concat(pred_res)
	else:
		top_nbs = [5, 10]
		thresholds = np.arange(0,0.5,0.2)
		test_y = [n_hot_decoder(i, label_set) for i in test_y]
	top_recall_f1 = (0,None,None)
	top_f1 = (0,None,None)

	for top_nb in top_nbs:
		for threshold in thresholds:
			decode_res = decode(pred_res, label_set, top_nb, threshold)
			precision = 0
			recall = 0
			test_len = 0.0
			pred_len = 0.0
	
			n1 = 0
			n2 = 0
			for i in range(data_set_size):
				test_set = set(test_y[i])
				pred_set = set(decode_res[i])
				#print(test_set)

				test_len += len(test_set)
				pred_len += len(pred_set)
				
				if len(pred_set) == 0 and len(test_set) == 0:
					precision += 1.0
					recall += 1.0
					continue
				if len(test_set) == 0 or len(pred_set)== 0:
					continue

				common_set = test_set & pred_set
				cnt = len(common_set)
				precision += float(cnt) / len(pred_set)
				recall += float(cnt) / len(test_set)

			precision /= data_set_size
			recall /= data_set_size
			test_len /= data_set_size
			pred_len /= data_set_size
			f1 = 2 * precision * recall / (precision + recall)
			if recall+f1 > top_recall_f1[0]:
				top_recall_f1 = (recall+f1, top_nb, threshold)
			if f1 > top_f1[0]:
				top_f1 = (f1, top_nb, threshold)
			print('(top %s, threshold %s) Precision: %.3lf' % (top_nb, threshold, precision))
			print('(top %s, threshold %s) Recall: %.3lf' % (top_nb, threshold, recall))
			print('(top %s, threshold %s) f1: %.3lf' % (top_nb, threshold, f1))
			print('(top %s, threshold %s) test_len %.2lf, pred_len %.2lf\n' % (top_nb, threshold, test_len, pred_len))			
	print('top (recall+f1)', top_recall_f1)
	print('top f1', top_f1)

def train(model, batch_size, nb_epoch, save_dir, train_X, train_y, val_data):
	print('train_X shape:', train_X.shape)
	print(train_X.shape[0], 'train samples')
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)

	start_time = time.time()
	save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.3f}.h5'
	check_pointer = ModelCheckpoint(save_path, 
        		save_best_only=True)
	history = model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=nb_epoch, 
		validation_data=val_data,
		validation_split=0.1, 
		callbacks=[check_pointer],
		)
	#model.save(save_path)
	plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.png')
	print('Training time(h):', (time.time()-start_time) / 3600)


def main():
	batch_size = 150
	nb_epoch = 100
	vector_size = 128
	max_sentence_len = 512
	min_count = 5
	max_sentence_nb = 10
	maxlen = 128

	save_dir = '../30w_weights/lr_' + str(datetime.now()).split('.')[0].split()[0] + '/' # model is saved corresponding to the datetime
	train_data_path = '../train_data/economic_fraud_train'
	val_data_path = '../train_data/economic_fraud_val'
	test_data_path = '../test_data/economic_fraud_test'
	label_data_path = '../train_data/economic_fraud_label'
	choice = "char"
	stopwords_path = '../train_data/law_stopwords.txt'
	words_path = '../dict/words_30w.txt'
	bin_path = '../dict/vector_30w.bin'
	input2idx_path = "../dict/30w_" + choice + '_input2idx.txt'
	label2idx_path = "../dict/30w_" + choice + '_label2idx.txt'
	weights_file_path = '../30w_weights/lr_2017-04-18/weights.16-0.116.h5'
	#weights_file_path = '../model_30w/gru95_2017-01-05/weights.28-0.047.h5'
	stopwords = ['本院认为','本院','认为','本','了','的','，'] 

	'''
	input_set, input2idx = get_input_set(label_data_path,choice,min_count)
	label_set, label2idx = get_label_set(label_data_path)
	input_set_size, label_set_size = len(input_set), [len(i) for i in label_set]
	print('input size:',input_set_size)
	print('label_size:',label_set_size, sum(label_set_size))
	
	print("saving idx...")
	save_input2idx(input2idx, input2idx_path)
	save_label2idx(label2idx, label2idx_path)
	'''
	print("loading idx...")
	input_set,input2idx = load_input2idx(input2idx_path)
	label_set,label2idx = load_label2idx(label2idx_path)
	input_set_size, label_set_size = len(input_set), [len(i) for i in label_set]	
	print('input size:',input_set_size)
	print('label size:',label_set_size, sum(label_set_size))
	#'''  

	'''
	model = build_logistic(input_set_size, label_set_size)
	#model = build_mlp(input_set_size, label_set_size)
    
	train_x = load_X_bag_of_words(train_data_path, max_sentence_len, input_set,stopwords)
	val_x = load_X_bag_of_words(val_data_path, max_sentence_len, input_set,stopwords)
	#train_x = load_X_hierarchical(train_data_path, max_sentence_nb,maxlen, input2idx,choice)
	#val_x = load_X_hierarchical(val_data_path, max_sentence_nb,maxlen, input2idx,choice)
	train_y = load_Y(train_data_path, label_set_size, label2idx)
	val_y = load_Y(val_data_path, label_set_size, label2idx)
	val_data = (val_x, val_y)
	train(model, batch_size, nb_epoch, save_dir, train_x, train_y, val_data)
	'''
    
	model = load_model(weights_file_path)
	test_x = load_X_bag_of_words(test_data_path, max_sentence_len, input_set, choice)
	#test_x = load_X_hierarchical(test_data_path, max_sentence_nb, maxlen, input2idx, stopwords)
	test_y = load_Y(test_data_path, label_set_size, label2idx)
	#print(len(test_x))
	test(model, test_x, test_y, label_set)


if __name__ == '__main__':
	main()
