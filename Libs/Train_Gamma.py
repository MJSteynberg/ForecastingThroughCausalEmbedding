"""
Class to wrap all the tensorflow/keras details for training Gamma
"""


import numpy as np 
import tensorflow as tf  ## USING v 2.0
from tensorflow import keras

class Train_NN_PCA:
    def __init__(self,X_train,Y_train, verbose=True,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=128):
        l,dim_x=X_train.shape
        l,dim_y=Y_train.shape
        
        self.dim_x=dim_x

        # Principal components

        X_pca=X_train[:,0:dim_x//2]
        U,Sig,Wt=np.linalg.svd(X_pca,full_matrices=True)
        T=U[:,:dim_x//2]*Sig
        self.W=Wt.T  #Principal component matrix

        T_train=np.zeros((l,dim_x))
        T_train[:,0:(dim_x//2)]=T
        T_train[:,(dim_x//2):dim_x]=X_train[:,(dim_x//2):dim_x]@self.W

        EPOCHS=epochs
        BATCH_SIZE=batch_size
        VERBOSE=verbose
        VALIDATION_SPLIT=0.2 

        # Defining the structure of the feed forward neural network using keras.

        self.model =tf.keras.Sequential()
        self.model.add(keras.layers.Dense(layer_dimension, input_shape=(dim_x,), name='input_layer', activation='relu'))
        for i in range(hidden_layers):
            self.model.add(keras.layers.Dense(layer_dimension, name='hidden_layer_'+str(i), activation='relu'))
        self.model.add(keras.layers.Dense(dim_y, name='output_layer', activation='tanh'))

        if VERBOSE:
            self.model.summary()

        # Training with different learning rates for the Adam Optimizer
        
        opt=keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss='MSE')
        self.model.fit(T_train,Y_train, batch_size=BATCH_SIZE,epochs=EPOCHS, 
                verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


        opt=keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='MSE')
        self.model.fit(T_train,Y_train, batch_size=BATCH_SIZE,epochs=EPOCHS, 
                verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


        opt=keras.optimizers.Adam(learning_rate=0.00001)
        self.model.compile(optimizer=opt, loss='MSE')
        self.model.fit(T_train,Y_train, batch_size=BATCH_SIZE,epochs=EPOCHS, 
                verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


    def predict(self,x):
        dim_x=self.dim_x
        x_=np.zeros((1,dim_x))
        x_[0]=x
        t_=np.zeros((1,dim_x))
        t_[0,0:(dim_x//2)]=x_[0,0:(dim_x//2)]@self.W
        t_[0,(dim_x//2):dim_x]=x_[0,(dim_x//2):dim_x]@self.W

        return self.model.predict(t_)[0]

    def principal_components(self,X_array):
        return X_array@self.W