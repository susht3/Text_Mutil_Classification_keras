import sys
sys.path.append('../main')
import keras
import numpy as np
from util94 import concat,flatten,n_hot_decoder,load_input2idx, load_label2idx,get_input_set,get_label_set
from util94 import load_text,load_text_hierarchical,get_stopwords,remove_stopwords
stopwords_path = '../train_data/law_stopwords.txt'
path_30w = '../train_data2/economic_legal'
stopwords =  get_stopwords(stopwords_path)
input_set_30w, input2idx_30w = get_input_set(path_30w,3,stopwords)
label_set_30w, label2idx_30w = get_label_set(path_30w)


class base(object):
	def __init__(self):
		self.choice = 'char'
		self.input_set_size, self.label_set_size = len(self.input_set), [len(i) for i in self.label_set]

	def decode(self, pred_res, with_prob):
		if type(pred_res) is list:
			ret = [flatten([n_hot_decoder(j, self.label_set[idx], self.top_nb[idx],
				self.threshold[idx], with_prob) for idx,j in enumerate(i)]) for i in pred_res]
		else:
			ret = [n_hot_decoder(i, self.label_set, self.top_nb, self.threshold,
				with_prob) for i in pred_res]
		return ret


	def match(self, x):
		if self.x_type == 'hierarchical':
			self.input_text = [load_text_hierarchical(i, self.max_sentence_nb,self.maxlen, self.input2idx, self.choice) for i in x ]
		else :
			self.input_text = [load_text(i, self.max_sentence_len, self.input2idx, self.choice) for i in x ]
		input_text = np.asarray(self.input_text)
		res = self.model.predict(input_text)
		res = concat(res)
		res = self.decode(res, True)
		return res
	

class hierarchical_30w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_nb = 10
		self.maxlen = 128
		self.x_type = 'hierarchical'
		self.input_set, self.input2idx = input_set_30w, input2idx_30w
		self.label_set, self.label2idx = label_set_30w, label2idx_30w
		self.weights_file_path = '../model_6w/hi_2016-12-27/weights.46-0.172.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

# 0.862
class gru_9w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.x_type = 'no'
		self.input_set, self.input2idx = input_set_30w, input2idx_30w
		self.label_set, self.label2idx = label_set_30w, label2idx_30w
		self.weights_file_path = '../model_9w/gru_2017-01-11/weights.10-0.067.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)


if __name__ == '__main__':
	
	input_text2 = ['本院认为，被告人张某某违反国家烟草专卖管理法律法规，未经烟草专卖行政主管部门许可，非法经营烟草专卖品，扰乱市场秩序，情节严重，其行为已构成规定的非法经营罪，西安市未央区人民检察院指控被告人犯非法经营罪的事实成立，依法应予惩处。在庭审中，被告人张某某如实供述自己的犯罪事实，并表示自愿认罪，可酌情从轻处罚。同时，经社区矫正机构调查评估，张某某符合社区矫正条件。依照之规定，']
	m2 = gru_9w()
	p2 = m2.match(input_text2)
	#print("p2:",p2)
