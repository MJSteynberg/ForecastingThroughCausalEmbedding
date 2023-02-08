"""
NOTE: TENSORFLOW >=2.0 NEEDS TO BE INSTALLED TO RUN THIS
The experiments were run using tensorflow version 2.4.1 and python 3.8.8

This script imports from the folder Libs: Net and Trainer_PCA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import seaborn as sns; sns.set_theme(style='white')
from scipy import stats
from scipy import signal
 
if input('Regenerate data? (y/n) DEFAULT = n \n')=='y':
##### Load Data
    print('REGENERATING...\n')
    t_delay=10#Takens delay

    def load_data(): 
        _data=np.loadtxt('Data/dp_data.txt')[:,1] #Use the y-coordinate of the second head
        
        
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


    #############################################
    # Learning Takens Map
    #############################################

    train_len=5000    # Nuber of datasteps used in training
    prediction_len=10500  #Timesteps predicted into the future after training
    discard=1000    #Wait for the variables to move to the attractor
    

    u_data=data[0:]

    X_train = data[discard: discard+train_len]
    Y_train = np.zeros((train_len,1))
    Y_train[:,0]=data[discard+t_delay:discard+train_len+t_delay,0]

    ### Traning the Takens map
    from Libs.Train_Takens import Train_NN_PCA
    trainer=Train_NN_PCA(X_train,Y_train,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=128)

    def F(x):  # The Learnt Takens Delay Map
        y=np.zeros(t_delay)
        for i in range(t_delay-1):
            y[i]=x[i+1]
            
        y[t_delay-1]=trainer.predict(x)
        return y

    #############################################
    # Prediction
    #############################################

    u_predicted=np.zeros((prediction_len,u_data.shape[1]))
    u_predicted[0]=u_data[train_len+discard]

    for t in range(prediction_len-1):
        print("Predicted {} out of {} \r".format(t,prediction_len),end='')
        u_predicted[t+1]=F(u_predicted[t])

    #############################################
    # Saving Data
    #############################################
    np.savetxt('Predicted_Data/DP/takens_delay_predicted.txt',u_predicted)
    np.savetxt('Predicted_Data/DP/takens_delay_actual.txt',u_data[train_len+discard:train_len+discard+prediction_len])


#############################################
# Plotting + Results
#############################################

# Wasserstein distance between prdicted and actual distributions

u_predicted=np.loadtxt('Predicted_Data/DP/takens_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/DP/takens_delay_actual.txt')

print(f"{len(u_predicted)}, {len(u_data)}")
S=0
plot_len1=1000
plot_len2=10000
pl_len3=10000

plt.rcParams['figure.figsize'] = [8, 2.5]

spec = gridspec.GridSpec( nrows=2,ncols=1,
                         hspace=0.3,height_ratios=[1, 1])
fig=plt.figure()
axs0=fig.add_subplot(spec[0])
axs1=fig.add_subplot(spec[1])

axs1.plot(range(S,S+plot_len1),u_predicted[S:S+plot_len1,0],lw=0.5,color='darkblue')
axs0.plot(range(S,S+plot_len1),u_data[S:S+plot_len1,0],lw=0.5,color='red')
axs0.xaxis.set_ticklabels([])
axs1.set_title('Predicted',fontsize=15)
axs0.set_title('Actual',fontsize=15)
plt.savefig('Img/DP/Takens-Traj.png')
plt.show()
### Recur
plt.rcParams['figure.figsize'] = [8, 4]

spec = gridspec.GridSpec( nrows=1,ncols=2,
                          hspace=0.3,height_ratios=[1])
fig=plt.figure()
axs2=fig.add_subplot(spec[0])
axs3=fig.add_subplot(spec[1])

axs2.plot(u_data[S:S+pl_len3, 0],u_data[S+16:S+pl_len3+16,0],lw=0.06,color='red')
axs2.set_title('Actual',fontsize=15)
axs3.plot(u_predicted[S:S+pl_len3, 0],u_predicted[S+16:S+pl_len3+16,0],lw=0.06,color='blue')
axs3.set_title('Predicted',fontsize=15)
plt.savefig('Img/DP/Takens-Recur.png')
plt.show()
####
#### Recur Overlay
plt.rcParams['figure.figsize'] = [5,5]


fig=plt.figure()
axs2=fig.add_subplot()


axs2.plot(u_data[S:S+pl_len3, 0],u_data[S+8:S+pl_len3+8,0],lw=0.06,color='red')

axs2.plot(u_predicted[S:S+pl_len3, 0],u_predicted[S+8:S+pl_len3+8,0],lw=0.06,color='blue')
axs2.xaxis.set_ticklabels([])
axs2.yaxis.set_ticklabels([])
axs2.set(xlabel = r'$y(t)$', ylabel = r'$y(t+8)$')
plt.savefig('Img/DP/Takens-Recur-Overlay_8.png')
plt.show()
####
#### Recur Overlay
plt.rcParams['figure.figsize'] = [5,5]


fig=plt.figure()
axs2=fig.add_subplot()


axs2.plot(u_data[S:S+pl_len3, 0],u_data[S+16:S+pl_len3+16,0],lw=0.06,color='red')

axs2.plot(u_predicted[S:S+pl_len3, 0],u_predicted[S+16:S+pl_len3+16,0],lw=0.06,color='blue')
axs2.xaxis.set_ticklabels([])
axs2.yaxis.set_ticklabels([])
axs2.set(xlabel = r'$y(t)$', ylabel = r'$y(t+16)$')
plt.savefig('Img/DP/Takens-Recur-Overlay_16.png')
plt.show()
####
plt.rcParams['figure.figsize'] = [8,4]
spec = gridspec.GridSpec( nrows=1,ncols=2,
                          wspace=0, width_ratios=[1, 1])
fig = plt.figure()
ax0 = fig.add_subplot(spec[1], projection='3d')


ax0.plot(u_predicted[:pl_len3,0], u_predicted[8:pl_len3+8,0],u_predicted[16:pl_len3+16,0],lw=0.05,color='blue')
ax0.set_xlabel(r'$y(t)$',fontsize=15)
ax0.set_ylabel(r"$y(t+8)$",fontsize=15)

ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.zaxis.set_ticklabels([])
ax0.set_ylim([-0.4,0.4])


ax1 = fig.add_subplot(spec[0], projection='3d')
ax1.plot(u_data[:pl_len3,0], u_data[8:pl_len3+8,0],u_data[16:pl_len3+16,0],lw=0.05,color='red')
ax1.set_xlabel(r'$y(t)$',fontsize=15)
ax1.set_ylabel(r"$y(t+8)$",fontsize=15)
ax0.set_zlabel(r"$y(t+16)$",fontsize=15)
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.zaxis.set_ticklabels([])
ax1.set_ylim([-0.4,0.4])

plt.savefig('Img/DP/Takens-Attractor.png')
plt.show()

## Attractor Overlay##
plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax0 = fig.add_subplot(projection='3d')


ax0.plot(u_predicted[:pl_len3,0], u_predicted[8:pl_len3+8,0],u_predicted[16:pl_len3+16,0],lw=0.1,color='blue')


ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.set_ylim([-0.4,0.4])

ax0.plot(u_data[:pl_len3,0], u_data[8:pl_len3+8,0],u_data[16:pl_len3+16,0],lw=0.070,color='red')

ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.zaxis.set_ticklabels([])
ax0.set_ylim([-0.4,0.4])



plt.savefig('Img/DP/Takens-Attractor-Overlay.png')
plt.show()



## Attractor Overlay With lables##
plt.rcParams['figure.figsize'] = [5,5]
fig = plt.figure()
ax0 = fig.add_subplot(projection='3d')


ax0.plot(u_predicted[:pl_len3,0], u_predicted[8:pl_len3+8,0],u_predicted[16:pl_len3+16,0],lw=0.1,color='blue')


ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.set_ylim([-0.4,0.4])

ax0.plot(u_data[:pl_len3,0], u_data[8:pl_len3+8,0],u_data[16:pl_len3+16,0],lw=0.070,color='red')

ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.zaxis.set_ticklabels([])
ax0.set_ylim([-0.4,0.4])
ax0.set_xlabel(r'$y(t)$',fontsize=12)
ax0.set_ylabel(r"$y(t+8)$",fontsize=12)
ax0.set_zlabel(r"$y(t+16)$",fontsize=12)
plt.savefig('Img/DP/Takens-Attractor-Overlay-Labels.png')
plt.show()

## Histogram 
plt.rcParams['figure.figsize'] = [8,4]
fig, axs=plt.subplots(1,1)
axs.hist(u_data[0:pl_len3,0],500,color='red',alpha=0.7, histtype = 'stepfilled')
axs.hist(u_predicted[0:pl_len3,0],500,color='darkblue',alpha=0.7, histtype = 'stepfilled')
axs.legend(['Actual','Predicted'], loc='upper center',fontsize=15)

axs.tick_params(axis='x', labelsize=22)
axs.tick_params(axis='y', labelsize=22)
plt.savefig('Img/DP/Takens-Hist.png')
plt.show()
## Autocorrelation

z_auto_pred = signal.correlate(u_predicted[0:pl_len3,0], u_predicted[0:pl_len3,0])
z_auto_data = signal.correlate(u_data[0:pl_len3,0], u_data[0:pl_len3,0])
z_cross = signal.correlate(u_data[0:pl_len3,0],u_predicted[0:pl_len3,0])
lags = signal.correlation_lags(len(u_data[0:pl_len3,0]), len(u_predicted[0:pl_len3,0]))

fig, (ax_orig, ax_noise) = plt.subplots(2, 1, sharex=True)
ax_noise.plot(lags[pl_len3-500:pl_len3+500], z_auto_data[pl_len3-500:pl_len3+500],lw=0.8,color='red')
ax_noise.set_title('Autocorrelation of Actual Data', fontsize = 15)

ax_orig.plot(lags[pl_len3-500:pl_len3+500], z_auto_pred[pl_len3-500:pl_len3+500],lw=0.8,color='blue')
ax_orig.set_title('Autocorrelation of Prediction ', fontsize = 15)

ax_orig.margins(0, 0.1)
fig.tight_layout()

plt.savefig('Img/DP/Takens-Autocor.png')
plt.show()

## Autocor overlay ##
fig, ax_orig = plt.subplots(1, 1, sharex=True)
ax_orig.set(ylim = (-150, 200))
ax_orig.plot(lags[pl_len3-500:pl_len3+500], z_auto_data[pl_len3-500:pl_len3+500],lw=1,color='red', label = "Actual")


ax_orig.plot(lags[pl_len3-500:pl_len3+500], z_auto_pred[pl_len3-500:pl_len3+500],lw=0.5,color='blue', label = 'Predicted')
#ax_orig.set_title('Autocorrelation', fontsize = 18)
ax_orig.legend(loc = 'upper center', fontsize = 22, ncol = 2)
ax_orig.xaxis.set_tick_params(labelsize = 22)
ax_orig.yaxis.set_tick_params(labelsize = 22)

ax_orig.margins(0, 0.1)
fig.tight_layout()


plt.savefig('Img/DP/Takens-Autocor-Overlay.png')
plt.show()

## Wasserstein Distances

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
ax.plot(x,y1,color='red',alpha=0.8)
ax.plot(x,y2,color='blue',alpha=0.8)
ax.set_title("")
ax.legend(["Actual","Predicted"])
ax.set_xlim([-0.8,0.8])

plt.savefig('Img/DP/Takens-CDF.png')
plt.show()