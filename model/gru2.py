from keras.layers import Input, Dense, Dropout, Flatten, merge, Reshape,Embedding
from keras.models import Model
from keras.layers import LSTM,TimeDistributed,Bidirectional, Highway,Activation,Convolution1D, MaxPooling1D,GRU,AveragePooling1D
#SimpleDeepRNN
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

def build_cnn_bgru(input_set_size, width, height, mul_nb_classes):
	print('cnn-gru model building...')
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
	conv3_1 = Convolution1D(160, 2, border_mode='same', activation='relu')(drop2)
	bn3 = BatchNormalization(mode=1)(conv3_1)
	#pool3 = MaxPooling1D(pool_length=2)(bn3)
	drop3 = Dropout(0.2)(bn3)
	
	#b = merge([bn4, drop3], mode='concat')
	bgru = Bidirectional(GRU(256,return_sequences=False), merge_mode='sum')(drop3)
	#gru = AveragePooling1D(pool_length=2)(bgru)
	drop = Dropout(0.5)(bgru)

	drop_3d = Reshape((256, 1))(drop)
	att = TimeDistributed(Dense(1))(drop_3d)
	att = Activation(activation="softmax")(att)
	merg = Flatten()(merge([drop_3d, att], mode='mul'))	
	
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
	#sgd = SGD(lr=0.01, momentum=0.9, decay=0.1, nesterov=False)
	model.compile(loss='binary_crossentropy',
                                #optimizer = 'sgd',  
				optimizer='adam',
				metrics=['accuracy'],
                                  )
	print("cnn-gru model has built.")
	return model

