from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape,Embedding,RepeatVector,Permute,AveragePooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.layers import LSTM,TimeDistributed,Bidirectional, Highway,Activation,Convolution1D, MaxPooling1D,GRU
#SimpleDeepRNN
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

def build_cnn_bgru_attention(input_set_size, width, height, mul_nb_classes):
	print('cnn-gru attention model building...')
	inputs = Input(shape=(height,),dtype='int32')
	embedd = Embedding(input_set_size, width, input_length=height)(inputs)
	
	# conv
	conv1_1 = Convolution1D(64, 3, border_mode='same', activation='relu')(embedd)
	bn1 = BatchNormalization(mode=1)(conv1_1)
	pool1 = MaxPooling1D(pool_length=2)(bn1)
	drop1 = Dropout(0.2)(pool1)
	
	# 2 conv
	conv2_1 = Convolution1D(128, 3, border_mode='same', activation='relu')(drop1)
	bn2 = BatchNormalization(mode=1)(conv2_1)
	pool2 = MaxPooling1D(pool_length=2)(bn2)
	drop2 = Dropout(0.2)(pool2)
	
	# 3 conv
	conv3_1 = Convolution1D(192, 2, border_mode='same', activation='relu')(drop2)
	bn3 = BatchNormalization(mode=1)(conv3_1)
	#pool3 = MaxPooling1D(pool_length=2)(bn3)
	drop3 = Dropout(0.1)(bn3)
	
	#b = merge([bn4, drop3], mode='concat')
	#blstm = Bidirectional(LSTM(256, return_sequences=False), merge_mode='sum')(drop3)
	gru = Bidirectional(GRU(256,return_sequences=True), merge_mode='sum')(drop3)
	drop = Dropout(0.5)(gru)

	'''
	drop_3d = Reshape((256, 1))(drop)
	att = TimeDistributed(Dense(1))(drop_3d)
	#att4 = Flatten()(att4)
	att = Activation(activation="softmax")(att)
	#att4 = RepeatVector(mul_nb_classes[3])(att4)
	#att4 = Permute((2,1))(att4)
	merg = Flatten()(merge([drop_3d, att], mode='mul'))
	'''
	# attention
	# attention
	mask = TimeDistributed(Dense(1))(drop) # compute the attention mask
	mask = Flatten()(mask)
	mask = Activation('softmax')(mask)
	mask = RepeatVector(256)(mask)
	mask = Permute([2, 1])(mask)

	# apply mask
	activations = merge([gru, mask], mode='mul')
	activations = GlobalAveragePooling1D()(activations)
	#activations = Flatten()(activations)

        # output
	out1 = Dense(mul_nb_classes[0], activation='sigmoid')(activations)
	merged1 = merge([out1, activations], mode='concat')
	out2 = Dense(mul_nb_classes[1], activation='sigmoid')(merged1)
	merged2 = merge([out2, activations], mode='concat')
	out3 = Dense(mul_nb_classes[2], activation='sigmoid')(merged2)
	merged3 = merge([out3, activations], mode='concat')
	out4 = Dense(mul_nb_classes[3], activation='sigmoid')(merged3)

	'''
	drop_3d = Reshape((mul_nb_classes[3],1))(drop)
	out4_3d = Reshape((mul_nb_classes[3], 1))(out4)
	att4_out4 = TimeDistributed(Dense(1))(out4_3d)
	att4_drop = TimeDistributed(Dense(1))(drop_3d)
	'''
	'''
	out4_3d = Reshape((mul_nb_classes[3], 1))(out4)
	att4 = TimeDistributed(Dense(1))(out4_3d)
	#att4 = Flatten()(att4)
	att4 = Activation(activation="softmax")(att4)
	#att4 = RepeatVector(mul_nb_classes[3])(att4)
	#att4 = Permute((2,1))(att4)
	merged4 = Flatten()(merge([out4_3d, att4], mode='mul'))
	'''
	out = [out1, out2, out3, out4]
	model = Model(input=[inputs], output=out)
	model.summary()
	sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)
	model.compile(loss='binary_crossentropy',
                                optimizer = sgd,  
				#optimizer='adam',
				metrics=['accuracy'],
                                  )
	print("cnn-gru attention model has built.")
	return model

