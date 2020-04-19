import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Input, Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.optimizers import Adam,  RMSprop

from keras.preprocessing.text import Tokenizer
from keras.models import Model

class NWModel(object):

    def __init__(self, *args, **kwargs):
        if args.__len__() ==1:
            self.xTrain, self.yTrain, self.xTest, self.yTest, self.category = \
                args[0][0],  args[0][1],  args[0][2],  args[0][3],  args[0][4] 
        else:
            self.xTrain, self.yTrain, self.xTest, self.yTest, self.category = \
                args[0], args[1], args[2], args[3], args[4] 

    def Dense_drop(self, layers_, n0, s0, d0):
        layers_= Dense(n0,  activation=s0)(layers_)
        return Dropout(d0)(layers_) if d0>0 else layers_

    def Dense_2(self, layers_, n0, s0, n1, s1):
        layers_= Dense(n0,  activation=s0)(layers_)
        return Dense(n1,  activation=s1)(layers_)

    def Dense_2_drop(self, layers_, n0, s0, n1, s1, d0):
        return Dropout(d0)(NWModel.Dense_2(self,  layers_, n0, s0, n1, s1))

    def Dense_3(self, layers_, n0, s0, n1, s1, n2, s2):
        layers_= Dense(n0,  activation=s0)(layers_)
        layers_ = Dense(n1,  activation=s1)(layers_)
        return Dense(n2,  activation=s2)(layers_)

    def Dense_3_drop(self, layers_, n0, s0, n1, s1, n2, s2, d0):
        return Dropout(d0)(NWModel.Dense_3(self,  layers_, n0, s0, n1, s1, n2, s2))
    
    def model_000(self): #Создаём полносвязную сеть
        inputs_basa = Input(shape=(self.xTrain.shape[1],), name="inputs_basa" ) 
        x = BatchNormalization() (inputs_basa)
        x = NWModel.Dense_drop(self, x, 3000, "relu", 0.2)
        x = NWModel.Dense_drop(self, x, 3000, "relu", 0.1)
        x = NWModel.Dense_3(self, x, 2000,  "sigmoid", 1000,  "relu", 500,  "sigmoid")
        x = Dense(100, activation="relu") (x)
        out_0 = Dense(len(self.category), activation='softmax')(x)
        model = Model(inputs=inputs_basa, outputs=out_0)
        NWModel.__fit(self, model, 40, 16) #70, 23

    def SetModelX(self, _func,  s = "ConfigAI.ws"):
        self.name_ws = s
        inputs_basa = Input(shape=(self.xTrain.shape[1],), name="inputs_basa" ) 
        x = BatchNormalization() (inputs_basa)
        x=_func(self, x)
        out_0 = Dense(len(self.category), activation='softmax')(x)
        model = Model(inputs=inputs_basa, outputs=out_0)
        model.compile(optimizer=RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        NWModel.__fit(self, model, 30, 16) #70, 23

    def LoadModelX(self, _func,  s = "ConfigAI.ws", bfit=True):
        self.name_ws = s
        inputs_basa = Input(shape=(self.xTrain.shape[1],), name="inputs_basa" ) 
        x = BatchNormalization() (inputs_basa)
        x=_func(self, x)
        out_0 = Dense(len(self.category), activation='softmax')(x)
        model = Model(inputs=inputs_basa, outputs=out_0)
        model.load_weights(self.name_ws)
        model.compile(optimizer=RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        if bfit:
            NWModel.__fit(self, model, 30, 16) #70, 23
            return
        else:
            return model

    def model_Embedding_0(self, maxWordsCount, xLen, matr):
        inputs_Embedding = Input(shape=(xLen,), name="inputs_Embedding" )
        xE = Embedding(maxWordsCount, matr, input_length=xLen) (inputs_Embedding)
        xE = SpatialDropout1D(0.2) (xE)
        xE = Flatten()(xE)
        xE = BatchNormalization() (xE)

#        xE = NWModel.Dense_3_drop(self, xE, 5000, "relu", 5000, "relu", 3000, "relu", 0.2)
#        xE = NWModel.Dense_3(self, xE, 2000, "relu", 1000, "relu", 1000, "relu")
#        xE = NWModel.Dense_2_drop(self, xE, 500,  "relu", 100,  "sigmoid", 0.1)

#        xE = NWModel.Dense_3_drop(self, xE, 2000, "relu", 1000, "relu", 3000, "relu", 0.1)
        xE = NWModel.Dense_3_drop(self, xE, 2000, "relu", 5000, "relu", 1000, "relu", 0.1)
#        xE = NWModel.Dense_2(self, xE, 1000, "relu", 500, "relu")
        xE = NWModel.Dense_2(self, xE, 100,  "relu", 50,  "relu")

        out_E=Dense(len(self.category), activation='softmax')(xE)
        modelE = Model(inputs=inputs_Embedding, outputs=out_E)
        modelE.compile(optimizer=RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        NWModel.__fit(self, modelE, 70, 21)

    def __plot(self, histor, acc, val_acc):
        import matplotlib.pyplot as plt 

        plt.plot(histor.history[acc], 
                 label='Доля верных ответов на обучающем наборе')
        plt.plot(histor.history[val_acc], 
                 label='Доля верных ответов на проверочном наборе')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()
        plt.show()

    def __fit(self, model, epochs, batch_size):
        history = model.fit(self.xTrain, 
                    self.yTrain, 
                    epochs=epochs, 
                    batch_size=batch_size,  #19
                    shuffle=True,
                    verbose=1,
                    validation_data=(self.xTest, self.yTest)
                    )
        model.save_weights(self.name_ws)

        self.__plot(history, 'accuracy', 'val_accuracy')
        return history
