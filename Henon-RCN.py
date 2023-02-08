"""
NOTE: TENSORFLOW >=2.0 NEEDS TO BE INSTALLED TO RUN THIS
The experiments were run using tensorflow version 2.4.1 and python 3.8.8

This script imports from the folder Libs: Net and Trainer_PCA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import seaborn as sns; sns.set_theme(style = 'white')
from scipy import stats
from scipy import signal

if input('Regenerate data? (y/n) DEFAULT = n \n')=='y':
##### Load Data
    print('REGENERATING...\n')

    t_delay=10 #RCN delay

    def load_data(): 
        ## Map:
        _data=np.loadtxt('Data/Henon_Map_J.txt')[:,0] #Use the first column of the data set
        
        data=np.zeros((_data.shape[0]-t_delay,t_delay))
        #Construct delay coordinates
        for i in range(t_delay):
            data[:,i]=_data[i:_data.shape[0]-t_delay+i]
        data = data-np.mean(data.flatten())
        data = data/(5*max(data.flatten()))
        rng=np.random.default_rng(12345)
        data+=rng.normal(0,.01,data.shape)
        
        return data

    data=load_data()

    ###################################
    # Plot Loaded Data:
    ###################################

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(data[:,0],data[:,1],'ko',markersize=0.1,color='red') #
    ax.set_title('Loaded Data:')
    plt.show()

    #############################################
    # Prediction
    #############################################

    train_len=5000    # Nuber of datasteps used in training
    prediction_len=10500  #Timesteps predicted into the future after training
    discard=1000    #Wait for the Network to forget
    dim=1000  #Dimension of the Network

    u_data=data

    X=np.zeros((train_len+discard+1,dim))

    rng=np.random.default_rng(12345)
    X[0]=rng.random(dim)

    from Libs.Net import Network
    net=Network(dim,alpha=0.9,a=0.5)

    # Driving the system
    for t in range(train_len+discard):
        print("Computed {} out of {} \r".format(t,train_len+discard),end='')    
        X[t+1]=net.g(u_data[t],X[t])
    print('')
            
    # Setting Up training Data: first {discard} datapoints are ignored
    Y_train=np.zeros((train_len,1))
    Y_train[:,0]=u_data[discard:discard+train_len,t_delay-1]   # Only need to map (x_n-1, x_n) -> u_n

    delay=2  ## Constructing a stack of {delay} time delayed vectors

    X_train=np.zeros((train_len,delay*dim))

    for i in range(delay):
        X_train[:,i*dim:(i+1)*dim]=X[discard-delay+i+1:discard-delay+i+1+train_len]

    ### Training Gamma. 
    from Libs.Train_Gamma import Train_NN_PCA
    trainer=Train_NN_PCA(X_train,Y_train,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=128)


    #############################################
    # Prediction
    #############################################

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

        for i in range(delay-1):
            X_stack[i*dim:(i+1)*dim]=X_stack[(i+1)*dim:(i+2)*dim]

        X_stack[(delay-1)*dim:delay*dim]=x_1
        
        u_new=np.zeros(t_delay)
        for i in range(t_delay-1):
            u_new[i]=u[i+1]
            
        u_new[t_delay-1]=trainer.predict(X_stack)
        u_predicted[t+1]=u_new

        x_0=x_1
        u=u_new

         
    #############################################
    # Saving Data
    #############################################
    np.savetxt('Predicted_Data/Henon/delay_comp_predicted.txt',u_predicted)
    np.savetxt('Predicted_Data/Henon/delay_comp_actual.txt',u_data[S:])

#############################################
# Plotting
#############################################

u_predicted=np.loadtxt('Predicted_Data/Henon/delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon/delay_comp_actual.txt')
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

axs1.plot(range(S,S+plot_len1),u_predicted[S:S+plot_len1,0],lw=0.5,color='darkblue')
axs0.plot(range(S,S+plot_len1),u_data[S:S+plot_len1,0],lw=0.5,color='red')
axs1.set_title('Predicted',fontsize=15)
axs0.set_title('Actual',fontsize=15)
plt.savefig('Img/Henon/RCN-Traj.png')
plt.show()
# 
plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax0 = fig.add_subplot()


ax0.plot(u_predicted[:pl_len3], u_predicted[1:pl_len3+1],'k.',color='blue',markersize=0.5)


ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])

ax0.plot(u_data[:pl_len3], u_data[1:pl_len3+1],'k.',color='red',markersize=0.5)

plt.savefig('Img/Henon/RCN-Attractor-Overlay.png')
plt.show()

#
####
plt.rcParams['figure.figsize'] = [16, 8]

spec = gridspec.GridSpec( nrows=1,ncols=2,
                         hspace=0.3,height_ratios=[1])
fig=plt.figure()
axs2=fig.add_subplot(spec[0])
axs3=fig.add_subplot(spec[1])

axs2.scatter(u_data[S:S+pl_len3, 0],u_data[S+1:S+pl_len3+1,0],lw=0.06,color='red', s = 0.5)
axs2.set_title('Actual',fontsize=15)
axs3.scatter(u_predicted[S:S+pl_len3, 0],u_predicted[S+1:S+pl_len3+1,0],lw=0.06,color='blue', s = 0.5)
axs3.set_title('Predicted',fontsize=15)
plt.savefig('Img/Henon/RCN-Attractor.png')
plt.show()

fig, axs=plt.subplots(1,1)
axs.hist(u_data[0:pl_len3,0],500,color='red',alpha=0.7, histtype = 'stepfilled')
axs.hist(u_predicted[0:pl_len3,0],500,color='darkblue',alpha=0.7, histtype = 'stepfilled')
axs.legend(['Actual','Predicted'], loc='upper center',fontsize=15)

axs.tick_params(axis='x', labelsize=22)
axs.tick_params(axis='y', labelsize=22)
plt.savefig('Img/Henon/RCN-Hist.png')
plt.show()

## Autocorrelation

z_auto_pred = signal.correlate(u_predicted[0:pl_len3,0], u_predicted[0:pl_len3,0])
z_auto_data = signal.correlate(u_data[0:pl_len3,0], u_data[0:pl_len3,0])
z_cross = signal.correlate(u_data[0:pl_len3,0],u_predicted[0:pl_len3,0])
# z_cross/=np.sqrt(z_auto_pred[pl_len3]*z_auto_data[pl_len3])
# z_auto_pred/=z_auto_pred[pl_len3]
# z_auto_data/=z_auto_data[pl_len3]
lags = signal.correlation_lags(len(u_data[0:pl_len3,0]), len(u_predicted[0:pl_len3,0]))

fig, (ax_orig, ax_noise) = plt.subplots(2, 1, sharex=True)
ax_noise.plot(lags[pl_len3-100:pl_len3+100], z_auto_data[pl_len3-100:pl_len3+100],lw=0.8,color='red')
ax_noise.set_title('Autocorrelation of Actual Data')

ax_orig.plot(lags[pl_len3-100:pl_len3+100], z_auto_pred[pl_len3-100:pl_len3+100],lw=0.8,color='blue')
ax_orig.set_title('Autocorrelation of Prediction ')

ax_orig.margins(0, 0.1)
fig.tight_layout()

plt.savefig('Img/Henon/RCN-Autocor.png')
plt.show()

fig, ax_orig = plt.subplots(1, 1, sharex=True)
ax_orig.plot(lags[pl_len3-500:pl_len3+500], z_auto_data[pl_len3-500:pl_len3+500],lw=0.8,color='red', label = "Actual")
ax_orig.set_title('Autocorrelation of Actual Data')
ax_orig.set(ylim=(-60, 180))
ax_orig.plot(lags[pl_len3-500:pl_len3+500], z_auto_pred[pl_len3-500:pl_len3+500],lw=0.8,color='blue', label="Predicted")
ax_orig.set_title('Autocorrelation of Prediction ')
ax_orig.legend(loc = 'upper center', fontsize = 22, ncol = 2)
ax_orig.xaxis.set_tick_params(labelsize = 22)
ax_orig.yaxis.set_tick_params(labelsize = 22)
ax_orig.margins(0, 0.1)
fig.tight_layout()

plt.savefig('Img/Henon/RCN-Autocor-Overlay.png')
plt.show()
### Wasserstein Distances

Ns=[10000]
for n in Ns:
    print(f"Wasserstein distance for {n} timesteps: ",stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0]) )

x_data=u_data[:,0]
x_data.sort()

x_predicted=u_predicted[:,0]
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
ax.plot(x,y1,color='red',alpha=0.5)
ax.plot(x,y2,color='blue',alpha=0.5)
ax.set_title("")
ax.legend(["Actual","Predicted"])
ax.set_xlim([-0.8,0.8])

plt.savefig('Img/Henon/RCN-CDF.png')
plt.show()