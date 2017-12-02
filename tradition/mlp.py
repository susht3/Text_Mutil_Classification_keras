import keras
from keras.layers import Input, Dense, Dropout, Flatten, merge,Activation
from keras.models import Model

def build_mlp(height,mul_nb_classes):
    print('mlp model building...')
    inputs = Input(shape=(height,),dtype='float32')
    dense1 = Dense(128)(inputs)
    dense2 = Dense(128)(dense1)
    
    out = [Dense(mul_nb_classes[i], activation='sigmoid')(dense2) for i in range(4)]
    model = Model(input=[inputs],output=out)
    model.summary()
    model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )
    print("mlp model has built.")
    return model

def test():
    return 

if __name__ == '__main__':
    model = build_mlp(128, 30)
    
    
