"""
NOTE: This script requires the scipy library

This script imports from the folder Libs: Net

This script uses data from the folder {Data} 

This script generates the table for the supplementary document: Table 1
"""

import numpy as np
import scipy.linalg as linalg

calc_len=5000    # Nuber of datasteps used in training
discard=1000    #Wait for the Network to forget

f = open("Pearson_Coefficients.txt", "w")

def corr_u(data):

    X_=np.zeros((calc_len,data.shape[1]))
    X_=data[discard-1:discard+calc_len-1]

    Y_=np.zeros((calc_len,data.shape[1]))
    Y_=data[discard:discard+calc_len]

    #Correlations

    Cov_X=1/(calc_len-1) * X_.T@X_
    Cov_Y=1/(calc_len-1) * Y_.T@Y_
    Cov_XY=1/(calc_len-1) * X_.T@Y_
    S=linalg.sqrtm(Cov_X@Cov_Y)
    Cor=np.trace(Cov_XY)/(np.trace(S))

    f.write(f"Multivariate correlation between u_n and u_n+1: {Cor}\n")

def corr_net(data, dim):

    f.write(f'Dimension of RNN: {dim} ')

    u_data=data[0:calc_len+discard]

    X=np.zeros((calc_len+discard+1,dim))

    rng=np.random.default_rng(12345)
    X[0]=rng.random(dim)

    from Libs.Net import Network
    net=Network(dim,alpha=0.99,a=0.5)

    # Driving the system
    for t in range(calc_len+discard):
        print("Computed {} out of {} \r".format(t,calc_len+discard),end='')    
        X[t+1]=net.g(u_data[t],X[t])
    print('')

    ##### Multivariate Corr between [x_{n-1},x_n] and [x_n, x_{n+1}]

    # Setting Up Data

    X_stack=np.zeros((calc_len-1,2*dim))
    X_stack[:,0:dim]=X[discard-1:discard+calc_len-2,:]
    X_stack[:,dim:2*dim]=X[discard:discard+calc_len-1,:]

    Y_stack=np.zeros((calc_len-1,2*dim))
    Y_stack[:,0:dim]=X[discard:discard+calc_len-1,:]
    Y_stack[:,dim:2*dim]=X[discard+1:discard+calc_len,:]

    # Correlations

    Cov_X=1/(calc_len-2) * X_stack.T@X_stack
    Cov_Y=1/(calc_len-2) * Y_stack.T@Y_stack
    Cov_XY=1/(calc_len-2) * X_stack.T@Y_stack
    S=linalg.sqrtm(Cov_X@Cov_Y)
    Cor=np.trace(Cov_XY)/(np.trace(S))

    f.write(f"Multivariate correlation between [x_n-1,x_n] and [x_n, x_n+1]: {Cor}\n")
    

f.write('\n=============================')
f.write("======= Double Pendulum ============")
f.write('=============================\n')

_data=np.loadtxt('Data/dp_data.txt')[:,1]

data=np.zeros((_data.shape[0],1))
data[:,0]=_data
data=data-np.mean(data)
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)


f.write('=============================')
f.write("====== Lorenz-Coarse ========")
f.write('=============================\n')

_data=np.loadtxt('Data/Lorenz_Coarse.txt')[:,0]
data=np.zeros((_data.shape[0],1))
data[:,0]=_data
data=data-np.mean(data)
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)


f.write('\n=============================')
f.write("==== Henon Map with Noise =====")
f.write('=============================\n')

_data=np.loadtxt('Data/Henon_Map_J.txt')[:,0]
data=np.zeros((_data.shape[0],1))
data[:,0]=_data
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)

f.write('\n=============================')
f.write("==== Henon Map with Less Noise =====")
f.write('=============================\n')

_data=np.loadtxt('Data/Henon_Map_J.txt')[:,0]
data[:,0]=_data
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)


f.write('\n=============================')
f.write("=========== PM ==============")
f.write('=============================\n')

_data=np.loadtxt('Data/PM_Map_06.txt')
data=np.zeros((_data.shape[0],1))
data[:,0]=_data
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)

f.write('\n=============================')
f.write("======= Logistic ============")
f.write('=============================\n')

_data=np.loadtxt('Data/Logistic_Map.txt')
data=np.zeros((_data.shape[0],1))
data[:,0]=_data
data = data/(max(data.flatten()))
rng=np.random.default_rng(12345)
data+=rng.normal(0,.05,data.shape)
data=data/5

corr_u(data)
corr_net(data,10)
corr_net(data,100)
corr_net(data,1000)

f.close()
