from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Model
from keras.layers import Activation, Dropout, Embedding,Bidirectional
from keras.layers import LSTM,TimeDistributed
from keras.layers.normalization import BatchNormalization

def build_blstm(input_set_size, width, height, mul_nb_classes):
	print("\nBLSTM Model Building ...")
	inputs = Input(shape=(height,), dtype='int32')
	embedd = Embedding(input_set_size, width, input_length=height)(inputs)
	
	# conv
	#conv1_1 = Convolution1D(128, 2, border_mode='same', activation='relu')(embedd)
	#bn1 = BatchNormalization(mode=1)(conv1_1)
	#pool1 = MaxPooling1D(pool_length=2)(bn1)
	#drop1 = Dropout(0.1)(pool1)
	
	
	blstm = Bidirectional(LSTM(256, return_sequences=False), merge_mode='sum')(embedd)
	drop = Dropout(0.5)(blstm)
	
	# output
	out1 = Dense(mul_nb_classes[0], activation='sigmoid')(drop)
	merged1 = merge([out1, drop], mode='concat')
	out2 = Dense(mul_nb_classes[1], activation='sigmoid')(merged1)
	merged2 = merge([out2, drop], mode='concat')
	out3 = Dense(mul_nb_classes[2], activation='sigmoid')(merged2)
	merged3 = merge([out3, drop], mode='concat')
	out4 = Dense(mul_nb_classes[3], activation='sigmoid')(merged3)

	out = [out1, out2, out3, out4]
	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'],
				  )
	print("BLSTM Model has built.")
	return model

def build_blstm_vec(input_set_size, width, height, mul_nb_classes,matrix):
	print("vec BLSTM Model Building ...")
	inputs = Input(shape=(height,), dtype='int32')
	#embedded = Embedding(input_set_size, width, input_length=height)(inputs)
	embedding_layer = Embedding(input_set_size+1,
                            width,
                            weights=[matrix],
                            input_length=height,
                            trainable=False) 
	embedded = embedding_layer(inputs)
	blstm = Bidirectional(LSTM(256, return_sequences=False), merge_mode='sum')(embedded)
	drop = Dropout(0.5)(blstm)

        # output
	out1 = Dense(mul_nb_classes[0], activation='sigmoid')(drop)
	merged1 = merge([out1, drop], mode='concat')
	out2 = Dense(mul_nb_classes[1], activation='sigmoid')(merged1)
	merged2 = merge([out2, drop], mode='concat')
	out3 = Dense(mul_nb_classes[2], activation='sigmoid')(merged2)
	merged3 = merge([out3, drop], mode='concat')
	out4 = Dense(mul_nb_classes[3], activation='sigmoid')(merged3)

	out = [out1, out2, out3, out4]
	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )
	print("vec BLSTM Model has built.")
	return model



def build_blstm_deep(input_set_size, width, height, mul_nb_classes):
	print("deep BLSTM Model Building ...")
	inputs = Input(shape=(height,), dtype='int32')
	embedded = Embedding(input_set_size, width, input_length=height)(inputs)
	#blstm = LSTM(128)(embedded)
	#blstm = Bidirectional(LSTM(128), merge_mode='sum')(embedded)

	blstm = Bidirectional(LSTM(128, return_sequences=False), merge_mode='sum')(embedded)
	#blstm2 = Bidirectional(LSTM(128, return_sequences=False), merge_mode='sum')(blstm)
	drop = Dropout(0.5)(blstm)

        # output
	out1 = Dense(mul_nb_classes[0], activation='sigmoid')(drop)
	merged1 = merge([out1, drop], mode='concat')
	out2 = Dense(mul_nb_classes[1], activation='sigmoid')(merged1)
	merged2 = merge([out2, drop], mode='concat')
	out3 = Dense(mul_nb_classes[2], activation='sigmoid')(merged2)
	merged3 = merge([out3, drop], mode='concat')
	out4 = Dense(mul_nb_classes[3], activation='sigmoid')(merged3)

	out = [out1, out2, out3, out4]
	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )
	print("deep BLSTM Model has built.")
	return model


def build_blstm_attention(input_set_size, width, height, mul_nb_classes):
	print("attention BLSTM Model Building ...")
	inputs = Input(shape=(height,), dtype='int32')
	embedd = Embedding(input_set_size, width, input_length=height)(inputs)
	
	# conv
	conv1_1 = Convolution1D(128, 2, border_mode='same', activation='relu')(embedd)
	bn1 = BatchNormalization(mode=1)(conv1_1)
	pool1 = MaxPooling1D(pool_length=2)(bn1)
	drop1 = Dropout(0.1)(pool1)

	blstm = Bidirectional(LSTM(128, return_sequences=False), merge_mode='sum')(drop1)
	drop = Dropout(0.5)(blstm)

	
	# output
	drop6_3d = Reshape((128,1))(drop) # TODO
        # output 1
	out1 = Dense(mul_nb_classes[0], activation='sigmoid')(drop)
        # output 2
	out1_3d = Reshape((mul_nb_classes[0], 1))(out1) # TODO
	att1_out1 = TimeDistributed(Dense(1))(out1_3d)
	att1_drop6 = TimeDistributed(Dense(1))(drop6_3d)
	merged1 = Flatten()(merge([att1_out1, att1_drop6], mode='concat', concat_axis=1))
	out2 = Dense(mul_nb_classes[1], activation='sigmoid')(merged1)
        # output 3
	out2_3d = Reshape((mul_nb_classes[1], 1))(out2) # TODO
	att2_out2 = TimeDistributed(Dense(1))(out2_3d)
	att2_drop6 = TimeDistributed(Dense(1))(drop6_3d)
	merged2 = Flatten()(merge([att2_out2, att2_drop6], mode='concat', concat_axis=1))
	out3 = Dense(mul_nb_classes[2], activation='sigmoid')(merged2)
        # output 4
	out3_3d = Reshape((mul_nb_classes[2], 1))(out3)
	att3_out3 = TimeDistributed(Dense(1))(out3_3d)
	att3_drop6 = TimeDistributed(Dense(1))(drop6_3d)
	merged3 = Flatten()(merge([att3_out3, att3_drop6], mode='concat', concat_axis=1))
	out4 = Dense(mul_nb_classes[3], activation='sigmoid')(merged3)
	out = [out1, out2, out3, out4]
	

	'''
	 # output
	bottle_neck_size = 150
	#out = range(len(mul_nb_classes))
	out0 = Dense(mul_nb_classes[0], activation='sigmoid')(drop)
	out = []
	out.append(out0)
	for i in range(1, len(mul_nb_classes)):
		merged = Reshape((bottle_neck_size+mul_nb_classes[i-1], 1))(merge([drop, out[i-1]], mode='concat'))
		sclice_a = Activation('softmax')(Flatten()(TimeDistributed(Dense(1, activation='tanh'))(merged)))
		sclice_b = Flatten()(merged)
		attention = merge([sclice_b, sclice_a], mode='mul')
		outi = Dense(mul_nb_classes[i], activation='sigmoid')(attention)
		out.append(outi)
	'''

	model = Model(input=[inputs], output=out)
	model.summary()
	model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )
	print("attention BLSTM Model has built.")
	return model

