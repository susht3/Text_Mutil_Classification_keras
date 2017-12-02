from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape, Activation, Permute
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional

def build_sentence_model(input_set_size,vector_size,maxlen,matrix):
	print('sentence model building...')
	word_inputs = Input(shape=(maxlen,), dtype='int32')
	embed = Embedding(input_set_size, vector_size, input_length=maxlen)(word_inputs)
	'''
	embed = Embedding(input_set_size, vector_size,
				weights=[matrix],
				input_length=maxlen,
				trainable=True)(word_inputs)
	'''

	# 1 conv
	#conv1_1 = Convolution1D(128, 3, border_mode='same', activation='relu')(embed)
	#bn1 = BatchNormalization(mode=1)(conv1_1)
	#pool1 = MaxPooling1D(pool_length=2)(bn1)
	#drop1 = Dropout(0.2)(pool1)
	
	'''
	# 2 conv
	conv2_1 = Convolution1D(128, 3, border_mode='same', activation='relu')(drop1)
	bn2 = BatchNormalization(mode=1)(conv2_1)
	pool2 = MaxPooling1D(pool_length=2)(bn2)
	drop2 = Dropout(0.2)(bn2)
	'''

        #gru
	bgru = Bidirectional(GRU(256,return_sequences=False), merge_mode='sum')(embed)
	drop = Dropout(0.5)(bgru)

	
	drop_3d = Reshape((256, 1))(drop)
	att = TimeDistributed(Dense(1))(drop_3d)
	att = Activation(activation="softmax")(att)
	merg1 = Flatten()(merge([drop_3d, att], mode='mul'))
	
	sentence_model = Model(input=[word_inputs], output=merg1)

	sentence_model.summary()

	'''
	sentence_model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  )
	'''
	print('sentence model built')
	return sentence_model


def build_hierarchical(input_set_size, vector_size, maxlen, max_sentence_nb, mul_label_set_size,matrix):	
	print('hierarchical model building...')

	sentence_input = Input(shape=(max_sentence_nb, maxlen,))
	sentence_model = build_sentence_model(input_set_size,vector_size,maxlen,matrix)
	paragraph = TimeDistributed(sentence_model)(sentence_input)
	
	gru = Bidirectional(GRU(128,return_sequences=False), merge_mode='sum')(paragraph)
	drop = Dropout(0.5)(gru)
	
	drop_3d = Reshape((128, 1))(drop)
	att = TimeDistributed(Dense(1))(drop_3d)
	att = Activation(activation="softmax")(att)
	merg2 = Flatten()(merge([drop_3d, att], mode='mul'))
	
	outs = []
	for i in range(0,4):
		if i == 0:
			merged = merg2#paragraph
		else:
			merged = merge([outs[i-1], merg2], mode='concat')
		#p = TimeDistributed(Dense(mul_label_set_size[i], activation='sigmoid'))(merged)
		#dense = Dense(256, activation='relu')(merged)
		p = Dense(mul_label_set_size[i], activation='sigmoid')(merged)
		outs.append(p)
	# out
	#outs = [GlobalMaxPooling1D()(outs[i]) for i in range(4)]
	
	#outs = [Dense(mul_label_set_size[i], activation='sigmoid')(drop5) for i in range(4)]
	model = Model(input=[sentence_input], output=outs)
	model.summary()
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  )

	print('hierarchical model built.')
	return model
