import sys
sys.path.append("../")
import os
import time
import json, pickle
import numpy as np
from datetime import datetime
from itertools import product
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from util2 import plot_loss_figure, load_X,load_Y,load_X_hierarchical, n_hot_decoder, unique
from util2 import get_input_set, list2str, get_label_set, flatten, concat
from util2 import save_input2idx, save_label2idx, load_input2idx, load_label2idx, load_matrix, train_vector
from blstm import build_blstm, build_blstm_deep,build_blstm_vec,build_blstm_attention
from clstm import  build_cnn_blstm
from gru import build_cnn_bgru
from clstm_vec import build_cnn_blstm_vec
from attention import build_cnn_bgru_attention
from fcn import build_fcn

# @profile
def decode(pred_res, label_set, top_nb, threshold, with_prob=False):
	if type(pred_res) is list:
		ret = [flatten([n_hot_decoder(j, label_set[idx], top_nb[idx], threshold[idx], with_prob) for idx,j in enumerate(i)]) for i in pred_res]
	else:
		ret = [n_hot_decoder(i, label_set, top_nb, threshold, with_prob) for i in pred_res]
	return ret


def set_of_hx(raw_set):
	ret = set()
	hx = [  "《中华人民共和国刑法》第七十二条","《中华人民共和国刑法》第七十三条",
		"《中华人民共和国刑法》第七十四条","《中华人民共和国刑法》第七十五条",
		"《中华人民共和国刑法》第七十六条","《中华人民共和国刑法》第七十七条"]
	for raw in raw_set:
		for h in hx:
			if h in raw:
				ret.add(raw)
				break
	return ret
			

# @profile
def test(model, test_X, test_y, label_set):
	data_set_size = len(test_X)
	print('data_set:'+str(data_set_size))
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
				
				test_set = set_of_hx(test_set)
				pred_set = set_of_hx(pred_set)
				if(len(test_set)>0):
					n1 = n1+1
				if(len(pred_set)>0):
					n2 = n2+1
				

				'''				
				print("test_set:")
				print(test_set)
				print(pred_set)
				'''
				test_len += len(test_set)
				pred_len += len(pred_set)
				
				#if len(pred_set) == 0:
				#	continue

				
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
			
			print('test text of hx:'+str(n1))	
			print('pred text of hx:'+str(n2))
			
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
	nb_epoch = 300
	vector_size = 128
	max_sentence_len = 512
	min_count = 1
	max_sentence_nb = 10
	maxlen = 128

	save_dir = '../model_6w/hi_' + str(datetime.now()).split('.')[0].split()[0] + '/' # model is saved corresponding to the datetime
	train_data_path = '../train_data/credit_card_fraud_train'
	val_data_path = '../train_data/credit_card_fraud_val'
	test_data_path = '../test_data/credit_card_fraud_test'
	label_data_path = '../train_data/credit_card_fraud_label'
	choice = "char"
	input2idx_path = "../dict/6w_" + choice + '_input2idx.txt'
	label2idx_path = "../dict/6w_" + choice + '_label2idx.txt'
	words_path = '../dict/words_6w.txt'
	bin_path = '../dict/vector_6w.bin'
	#weights_file_path = '../model_6w/cnn_bgru_2016-12-19/weights.23-0.171.h5'
	weights_file_path = '../model_6w/hi_2016-12-27/weights.46-0.172.h5'
	#weights_file_path = '../model_6w/blstm_2016-12-16/weights.24-0.193.h5'
	#weights_file_path = '../model_6w/blstm_attention_2016-12-16/weights.27-0.198.h5'
	
	
	input_set, input2idx = get_input_set(label_data_path,choice,min_count)
	label_set, label2idx = get_label_set(label_data_path)
	input_set_size, label_set_size = len(input_set), [len(i) for i in label_set]
	print('input size:',input_set_size)
	print('label_size:',label_set_size, sum(label_set_size))
	'''
	print("saving idx...")
	save_input2idx(input2idx, input2idx_path)
	save_label2idx(label2idx, label2idx_path)	
	
	print("loading idx...")
	input_set,input2idx = load_input2idx(input2idx_path)
	label_set,label2idx = load_label2idx(label2idx_path)
	input_set_size, label_set_size = len(input_set), [len(i) for i in label_set]	
	print('input size:',input_set_size)
	print('label size:',label_set_size, sum(label_set_size))
	
	print('training vectors...')
	#train_vector(input_set,vector_size,min_count,words_path,bin_path)
	matrix = load_matrix(bin_path, input2idx)
	'''
	
	#model = build_cnn_bgru_attention(input_set_size, vector_size, max_sentence_len, label_set_size)
	model = build_fcn(input_set_size, vector_size, max_sentence_len, label_set_size)
	#model = build_cnn_blstm(input_set_size, vector_size, max_sentence_len, label_set_size)
	#model = build_cnn_blstm_vec(input_set_size, vector_size, max_sentence_len, label_set_size, matrix)	
	
	train_x = load_X(train_data_path, max_sentence_len, input2idx,choice)
	train_y = load_Y(train_data_path, label_set_size, label2idx)
	val_x = load_X(val_data_path, max_sentence_len, input2idx,choice)
	val_y = load_Y(val_data_path, label_set_size, label2idx)
	val_data = (val_x, val_y)
	#model.load_weights(weights_file_path)
	train(model, batch_size, nb_epoch, save_dir, train_x, train_y, val_data)
	
	#model = load_model(weights_file_path)
	#test_x = load_X_hierarchical(test_data_path, max_sentence_nb,maxlen, input2idx, choice)
	test_x = load_X(test_data_path, max_sentence_nb,maxlen, input2idx, choice)
	test_y = load_Y(test_data_path, label_set_size, label2idx)
	print(len(test_x))
	print(len(test_y))
	test(model, test_x, test_y, label_set)
	
	
if __name__ == '__main__':
	main()
