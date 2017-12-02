import sys
sys.path.append('../main')
import keras
import numpy as np
from utils import load_text,concat,flatten,n_hot_decoder,load_input2idx, load_label2idx,load_X,load_Y,get_input_set,get_label_set

input_set_6w, input2idx_6w = get_input_set('../train_data/credit_card_fraud_label','char')
input_set_30w, input2idx_30w = get_input_set('../train_data/economic_fraud_label','char')

label_set_6w, label2idx_6w = get_label_set('../train_data/credit_card_fraud_label')
label_set_30w, label2idx_30w = get_label_set('../train_data/economic_fraud_label')


class base(object):
	def __init__(self):
		self.choice = 'char'
		self.input_set_size, self.label_set_size = len(self.input_set), [len(i) for i in self.label_set]

	def decode(self, pred_res, with_prob):
		print('pred_res:',pred_res)
		if type(pred_res) is list:
			ret = [flatten([n_hot_decoder(j, self.label_set[idx], self.top_nb[idx],
				self.threshold[idx], with_prob) for idx,j in enumerate(i)]) for i in pred_res]
		else:
			ret = [n_hot_decoder(i, self.label_set, self.top_nb, self.threshold,
				with_prob) for i in pred_res]
		print('\ndecode_res:\n',ret)
		return ret


	def match(self, x):
		input_text = [load_text(i, self.max_sentence_len, self.input2idx, self.choice) for i in x ]
		input_text = np.asarray(input_text)
		res = self.model.predict(input_text)
		res = concat(res)
		res = self.decode(res, True)
		return res

# 0.896
class blstm_6w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_6w, input2idx_6w
		self.label_set, self.label2idx = label_set_6w, label2idx_6w
		self.weights_file_path = '../model_6w/blstm_2016-12-16/weights.24-0.193.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

# 0.85
class blstm_30w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_30w, input2idx_30w
		self.label_set, self.label2idx = label_set_30w, label2idx_30w
		self.weights_file_path = '../model_30w/blstm_2016-12-15/weights.08-0.047.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.3, 0.3, 0.3, 0.3]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)


# 0.889
class blstm_attention_6w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_6w, input2idx_6w
		self.label_set, self.label2idx = label_set_6w, label2idx_6w
		self.weights_file_path = '../model_6w/blstm_attention_2016-12-16/weights.27-0.198.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.3, 0.3, 0.3, 0.3]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

# 0.848 
class blstm_attention_30w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_30w, input2idx_30w
		self.label_set, self.label2idx = label_set_30w, label2idx_30w

		self.weights_file_path = '../model_30w/blstm_attention_2016-12-16/weights.09-0.049.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.3, 0.3, 0.3, 0.3]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)


# 0.903
class cnn_blstm_6w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_6w, input2idx_6w
		self.label_set, self.label2idx = label_set_6w, label2idx_6w
		self.weights_file_path = '../model_6w/3cnn_blstm_2016-12-14/weights.37-0.177.h5'
		#self.weights_file_path = '../model_6w/3cnn_blstm_2016-12-14/weights.29-0.175.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

# 0.870
class cnn_blstm_30w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_30w, input2idx_30w
		self.label_set, self.label2idx = label_set_30w, label2idx_30w
		self.weights_file_path = '../model_30w/3cnn_blstm_2016-12-14/weights.39-0.042.h5'
		#self.weights_file_path = '../model_30w/cnn_blstm_4cnn_2016-12-13/weights.50-0.048.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

'''
# 0.906
class cnn_bgru_6w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_6w, input2idx_6w
		self.label_set, self.label2idx = label_set_6w, label2idx_6w
		self.weights_file_path = '../model_6w/cnn_bgru_2016-12-19/weights.23-0.171.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)

# 
class cnn_bgru_30w(base):
	def __init__(self):
		self.vector_size = 128
		self.max_sentence_len = 512
		self.input_set, self.input2idx = input_set_6w, input2idx_6w
		self.label_set, self.label2idx = label_set_6w, label2idx_6w
		self.weights_file_path = '../model_30w/cnn_bgru_2016-12-19/weights.26-0.043.h5'
		self.top_nb, self.threshold = [2, 10, 10, 6], [0.4, 0.4, 0.4, 0.4]
		base.__init__(self)
		self.model = keras.models.load_model(self.weights_file_path)
'''


if __name__ == '__main__':
	m = cnn_blstm_6w()
	lab = m.label_set
	input_text = ['本院认为，被告人杨某以非法占有为目的，盗窃他人信用卡并使用，取款人民币2500元，数额较大，其行为已构成盗窃罪。被告人杨某如实供述自己罪行，依法予以从轻处罚。被告人杨某退赔被害人损失并取得谅解，酌情从轻处罚。公诉机关指控被告人杨某犯盗窃罪的事实清楚，证据确实、充分，指控罪名成立。据此，依照及之规定，']
	#input_text = ['本院认为，被告人刘某以非法占有为目的，盗窃信用卡并使用，数额较大，其行为已触犯刑律，构成盗窃罪。公诉机关指控的罪名成立。被告人刘某归案后能如实供述自己罪行，已退赔，有悔罪表现，依法与酌情从轻处罚并适用缓刑。为严明国法，惩罚犯罪，保护公私财产权利不受侵犯，根据被告人犯罪的事实、犯罪的性质、情节和对于社会的危害程度，依照之规定，']
	
	p = m.match(input_text)
	

	'''	
	test_data_path = '../test_data/credit_card_fraud_test'	
	test_x = load_X(test_data_path, m.max_sentence_len, m.input2idx, m.choice)
	test_y = load_Y(test_data_path, m.label_set_size, m.label2idx)
	test(m.model,test_x, test_y, m.label_set)
	'''
	'''
	input_text2 = ['本院认为，被告人张某某违反国家烟草专卖管理法律法规，未经烟草专卖行政主管部门许可，非法经营烟草专卖品，扰乱市场秩序，情节严重，其行为已构成规定的非法经营罪，西安市未央区人民检察院指控被告人犯非法经营罪的事实成立，依法应予惩处。在庭审中，被告人张某某如实供述自己的犯罪事实，并表示自愿认罪，可酌情从轻处罚。同时，经社区矫正机构调查评估，张某某符合社区矫正条件。依照之规定，']
	m2 = cnn_blstm_30w()
	p2 = m2.match(input_text2)
	print("p2:",p2)
	'''
