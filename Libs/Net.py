"""
Class used to define the Network.
"""

import numpy as np 

class Network:
    def __init__(self, dimension, a=0.5, alpha=0.99,seed=12345):
        self._a=a
        self._alpha=alpha
        self._dim=dimension
        rng=np.random.default_rng(seed)# Pad with zeros

        ################################
        ## Generate Internal Matrices
        ################################

        #input matrix:
        self._W_in=(rng.random((self._dim,self._dim))-0.5)

        #Network Matrix:
        W=rng.random((self._dim,self._dim))-0.5
        self._W=1/max(abs(np.linalg.eigvals(W))) * W #Spectral Radius =1 

    def g(self,u,x):
        # Pad u with zeros
        u_=np.zeros(self._dim) 
        u_[0:len(u)]=u
        # W_in is a matrix with dimension (dim, dim), and u is padded with zeros. For the Echo State Network,
        # this is equivalent to W_in being a matrix of dimension (dim, 1) and u not being padded.
    
        return (1-self._a)*x +self._a*np.tanh(self._W_in@u_+self._alpha*self._W@x)



        