import keras
from keras.layers import Input, Dense, Dropout, Flatten, merge,Activation
from keras.models import Model

def build_logistic(height,mul_nb_classes):
    print('logistic model building...')
    inputs = Input(shape=(height,),dtype='float32')
    
    out = [Dense(mul_nb_classes[i], activation='sigmoid')(inputs) for i in range(4)]
    #out = Dense(nb_classes, activation='sigmoid')(inputs) 

    model = Model(input=[inputs],output=out)
    model.summary()
    model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )
    print("logistic model has built.")
    return model

def test():
    return 

if __name__ == '__main__':
    model = build_logistic(128, 30)
    
    
