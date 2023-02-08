"""
NOTE: TENSORFLOW >=2.0 NEEDS TO BE INSTALLED TO RUN THIS
The experiments were run using tensorflow version 2.4.1 and python 3.8.8

This script imports from the folder Libs: Net and Trainer_PCA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import seaborn as sns; sns.set_theme(style = "white")
from scipy import stats
from scipy import signal

if input('Regenerate data? (y/n) DEFAULT = n \n')=='y':
##### Load Data
    print('REGENERATING...\n')

    def load_data(): 
        _data=np.loadtxt('Data/PM_Map_06.txt')
        np.random.seed(500)

        data=np.zeros((_data.shape[0],1))
        data[:,0]=_data
        data=data-np.mean(data)
        data = data/(5*max(data.flatten()))
        data=data+np.random.normal(0,0.01,(_data.shape[0],1))

        return data

    data=load_data()

    ###################################
    # Plot Loaded Data:
    ###################################

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(data[0:len(data)-1,0],data[1:,0],'ko',markersize=0.1,color='red')
    ax.set_title('Loaded Data:')
    plt.show()

    #############################################
    # Prediction
    #############################################

    train_len=5000    # Nuber of datasteps used in training
    prediction_len=10500  #Timesteps predicted into the future after training
    discard=1000    #Wait for the Network to forget
    dim=1000  #Dimension of the Network

    u_data=data[0:]

    X=np.zeros((train_len+discard+1,dim))

    rng=np.random.default_rng(12345)
    X[0]=rng.random(dim)

    from Libs.Net import Network
    net=Network(dim,alpha=0.99,a=0.5)

    # Driving the system
    for t in range(train_len+discard):
        print("Computed {} out of {} \r".format(t,train_len+discard),end='')    
        X[t+1]=net.g(u_data[t],X[t])
    print('')
            
    # Setting Up training Data: first {discard} datapoints are ignored
    Y_train=u_data[discard:discard+train_len]

    ## Constructing a stack of {delay} time delayed vectors, 
    ## the Map Gamma from theory uses delay=2
    delay=2  

    X_train=np.zeros((train_len,delay*dim))

    for i in range(delay):
        X_train[:,i*dim:(i+1)*dim]=X[discard-delay+i+1:discard-delay+i+1+train_len]

    ### Training Gamma. 
    from Libs.Train_Gamma import Train_NN_PCA
    trainer=Train_NN_PCA(X_train,Y_train,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=128)

    # Setting up variables
    u_predicted=np.zeros((prediction_len,u_data.shape[1]))
    u=u_data[train_len+discard]
    x_0=X[train_len+discard]
    u_predicted[0]=u
    X_stack=np.zeros(delay*dim)

    for i in range(delay):
        X_stack[i*dim:(i+1)*dim]=X[train_len+discard-delay+i+1]

    S=train_len+discard

    # Running Prediction
    for t in range(prediction_len-1):
        print("Predicted {} out of {} \r".format(t,prediction_len),end='')
        x_1=net.g(u,x_0)
        # X[S+t+1]=x_1

        for i in range(delay-1):
            X_stack[i*dim:(i+1)*dim]=X_stack[(i+1)*dim:(i+2)*dim]

        X_stack[(delay-1)*dim:delay*dim]=x_1
        
        u=trainer.predict(X_stack)
        u_predicted[t+1]=u
        x_0=x_1
        
    #############################################
    # Saving Data
    #############################################
    np.savetxt('Predicted_Data/PM/simple_1d_predicted.txt',u_predicted)
    np.savetxt('Predicted_Data/PM/simple_1d_actual.txt',u_data[S:])

#############################################
# Plotting
#############################################

u_data=np.loadtxt('Predicted_Data/PM/simple_1d_actual.txt')
u_predicted=np.loadtxt('Predicted_Data/PM/simple_1d_predicted.txt')
print(f"{len(u_predicted)}, {len(u_data)}")
S=0
plot_len1=1000
plot_len2=10000
pl_len3=10000
plt.rcParams['figure.figsize'] = [15, 4]

spec = gridspec.GridSpec( nrows=2,ncols=1,
                         hspace=0.3,height_ratios=[1, 1])
fig=plt.figure()
axs0=fig.add_subplot(spec[0])
axs1=fig.add_subplot(spec[1])

axs1.plot(range(S,S+plot_len1),u_predicted[S:S+plot_len1],lw=0.5,color='darkblue')
axs0.plot(range(S,S+plot_len1),u_data[S:S+plot_len1],lw=0.5,color='red')
axs1.set_title('Predicted',fontsize=15)
axs0.set_title('Actual',fontsize=15)
plt.savefig('Img/PM/RCN-Traj.png')
plt.show()


plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax0 = fig.add_subplot()


ax0.plot(u_predicted[:pl_len3], u_predicted[1:pl_len3+1],'k.',color='blue',markersize=0.5)


ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])

ax0.plot(u_data[:pl_len3], u_data[1:pl_len3+1],'k.',color='red',markersize=0.5)


plt.savefig('Img/PM/RCN-Attractor-Overlay.png')
plt.show()


plt.rcParams['figure.figsize'] = [8,4]
spec = gridspec.GridSpec( nrows=1,ncols=2,
                         wspace=0, width_ratios=[1, 1])
fig = plt.figure()
ax0 = fig.add_subplot(spec[1])


ax0.plot(u_predicted[:pl_len3], u_predicted[1:pl_len3+1],'k.',color='blue',markersize=0.5)
ax0.set_xlabel(r'$u_x$',fontsize=25)
ax0.set_ylabel("",fontsize=25)

ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])


ax1 = fig.add_subplot(spec[0])
ax1.plot(u_data[:pl_len3], u_data[1:pl_len3+1],'k.',color='red',markersize=0.5)
ax1.set_xlabel(r'$u_x$',fontsize=25)
ax1.set_ylabel(r'$u_y$',fontsize=25)
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])

plt.savefig('Img/PM/RCN-Attractor.png')
plt.show()



fig, axs=plt.subplots(1,1)
axs.hist(u_data[0:pl_len3],500,color='red',alpha=0.7, histtype = 'stepfilled')
axs.hist(u_predicted[0:pl_len3],500,color='darkblue',alpha=0.7, histtype = 'stepfilled')
axs.legend(['Actual','Predicted'], loc='upper center',fontsize=15)

axs.tick_params(axis='x', labelsize=22)
axs.tick_params(axis='y', labelsize=22)
plt.savefig('Img/PM/RCN-Hist.png')
plt.show()

## Autocorrelation

z_auto_pred = signal.correlate(u_predicted[0:pl_len3], u_predicted[0:pl_len3])
z_auto_data = signal.correlate(u_data[0:pl_len3], u_data[0:pl_len3])
z_cross = signal.correlate(u_data[0:pl_len3],u_predicted[0:pl_len3])
# z_cross/=np.sqrt(z_auto_pred[pl_len3]*z_auto_data[pl_len3])
# z_auto_pred/=z_auto_pred[pl_len3]
# z_auto_data/=z_auto_data[pl_len3]
lags = signal.correlation_lags(len(u_data[0:pl_len3]), len(u_predicted[0:pl_len3]))

fig, (ax_orig, ax_noise) = plt.subplots(2, 1, sharex=True)
ax_noise.plot(lags[pl_len3-100:pl_len3+100], z_auto_data[pl_len3-100:pl_len3+100],lw=0.8,color='red')
ax_noise.set_title('Autocorrelation of Actual Data')

ax_orig.plot(lags[pl_len3-100:pl_len3+100], z_auto_pred[pl_len3-100:pl_len3+100],lw=0.8,color='blue')
ax_orig.set_title('Autocorrelation of Prediction ')

ax_orig.margins(0, 0.1)
fig.tight_layout()

plt.savefig('Img/PM/RCN-Autocor.png')
plt.show()

fig, ax = plt.subplots()

ax.plot(lags[pl_len3-500:pl_len3+500], z_auto_data[pl_len3-500:pl_len3+500],lw=0.8,color='red', label="Actual")

ax.plot(lags[pl_len3-500:pl_len3+500], z_auto_pred[pl_len3-500:pl_len3+500],lw=0.8,color='blue', label="Predicted")

ax.legend(loc = 'upper center', fontsize = 22, ncol = 2)
ax.xaxis.set_tick_params(labelsize = 22)
ax.yaxis.set_tick_params(labelsize = 22)
ax.margins(0, 0.1)
fig.tight_layout()

plt.savefig('Img/PM/RCN-Autocor-Overlay.png')
plt.show()

### Wasserstein Distances

Ns=[10000]
for n in Ns:
    print(f"Wasserstein distance for {n} timesteps: ",stats.wasserstein_distance(u_data[0:n],u_predicted[0:n]) )

x_data=u_data[:]
x_data.sort()

x_predicted=u_predicted[:]
x_predicted.sort()

x=np.linspace(-1,1,1000,endpoint=False)


def bin_search(x,sorted_data):
    l=0
    u=len(sorted_data)
    while l<u-1:
        m=(l+u)//2
        if x<sorted_data[m]:
            u=m
        else: l=m

    return l

def CDF(x, sorted_data):
    s=bin_search(x,sorted_data)
    return 1/len(sorted_data)*s


plt.rcParams['figure.figsize'] = [8, 4]
y1=[CDF(t,x_data) for t in x]
y2=[CDF(t,x_predicted) for t in x]
fig,ax=plt.subplots()
ax.plot(x,y1,color='red',alpha=0.8)
ax.plot(x,y2,color='blue',alpha=0.8)
ax.set_title("")
ax.legend(["Actual","Predicted"])
ax.set_xlim([-0.8,0.8])

plt.savefig('Img/PM/RCN-CDF.png')
plt.show()