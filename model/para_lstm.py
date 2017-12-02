from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape, Activation, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional


def build_para_lstm(input_set_size, width, height, mul_nb_classes):
	# input
	inputs = Input(shape=(height,), dtype='int32')
	embed = Embedding(input_set_size, width, input_length=height)(inputs)
	sentence_model = Bidirectional(LSTM(64, return_sequences=True))(embed)
	pool = GlobalMaxPooling1D()(sentence_model)
	# output
	out = [Dense(mul_nb_classes[i], activation='sigmoid')(pool) for i in range(4)]
	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  )

	return model