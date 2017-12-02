from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape, Activation, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional


def build_local(input_set_size, vector_size, maxlen,max_sentence_nb, mul_label_set_size):
	# sentence vector
	# input
	word_inputs = Input(shape=(maxlen,), dtype='int32')
	embed = Embedding(input_set_size, vector_size, input_length=maxlen)(word_inputs)
	# 1 conv
	conv1_1 = Convolution1D(64, 3, border_mode='same', activation='relu')(embed)
	bn1 = BatchNormalization(mode=1)(conv1_1)
	pool1 = MaxPooling1D(pool_length=2)(bn1)
	drop1 = Dropout(0.3)(pool1)
	# 2 conv
	conv2_1 = Convolution1D(128, 2, border_mode='same', activation='relu')(drop1)
	bn2 = BatchNormalization(mode=1)(conv2_1)
	pool2 = MaxPooling1D(pool_length=1)(bn2)
	drop2 = Dropout(0.3)(pool2)
	# 3 conv
	conv3_1 = Convolution1D(160, 2, border_mode='same', activation='relu')(drop2)
	bn3 = BatchNormalization(mode=1)(conv3_1)
	pool4 = GlobalMaxPooling1D()(bn3)
	drop4 = Dropout(0.5)(pool4)
	sentence_model = Model(input=[word_inputs], output=drop4)
	# paragraph vector
	sentence_input = Input(shape=(max_sentence_nb, maxlen,))
	paragraph = TimeDistributed(sentence_model)(sentence_input)
	paragraph_model = [TimeDistributed(Dense(mul_label_set_size[i], activation='sigmoid'))(paragraph) for i in range(4)]
	# out
	outs = [GlobalMaxPooling1D()(paragraph_model[i]) for i in range(4)]
	model = Model(input=[sentence_input], output=outs)
	model.summary()
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  )

	return model
