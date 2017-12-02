from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding


def build_para_cnn(input_set_size, vector_size, max_sentence_len, mul_nb_classes):
	# input
	inputs = Input(shape=(max_sentence_len,), dtype='int32')
	embed = Embedding(input_set_size, vector_size, input_length=max_sentence_len)(inputs)
	# 1 conv
	conv1_1 = Convolution1D(48, 2, border_mode='same', activation='relu')(embed)
	bn1 = BatchNormalization(mode=1)(conv1_1)
	pool1 = MaxPooling1D(pool_length=2)(bn1)
	drop1 = Dropout(0.5)(pool1)
	# 2 conv
	conv2_1 = Convolution1D(64, 2, border_mode='same', activation='relu')(drop1)
	bn2 = BatchNormalization(mode=1)(conv2_1)
	pool2 = MaxPooling1D(pool_length=2)(bn2)
	drop2 = Dropout(0.5)(pool2)
	# 3 conv
	conv3_1 = Convolution1D(128, 2, border_mode='same', activation='relu')(drop2)
	bn3 = BatchNormalization(mode=1)(conv3_1)
	pool3 = MaxPooling1D(pool_length=2)(bn3)
	drop3 = Dropout(0.5)(pool3)
	# 4 conv
	conv4_1 = Convolution1D(160, 2, border_mode='same', activation='relu')(drop3)
	bn4 = BatchNormalization(mode=1)(conv4_1)
	pool4 = MaxPooling1D(pool_length=2)(bn4)
	drop4 = Dropout(0.5)(pool4)
	# 5 conv
	conv5_1 = Convolution1D(192, 2, border_mode='same', activation='relu')(drop4)
	bn5 = BatchNormalization(mode=1)(conv5_1)
	pool5 = MaxPooling1D(pool_length=2)(bn5)
	drop5 = Dropout(0.5)(pool5)
	# flaten
	flat = Flatten()(drop5)
	# output
	out = [Dense(mul_nb_classes[i], activation='sigmoid')(flat) for i in range(4)]
	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  )

	return model