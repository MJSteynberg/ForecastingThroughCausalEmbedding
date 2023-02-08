"""
Program to demonstrate global dissipativity.

Differnt runs will generate different figures due to the training of gamma. However, 
global dissipativity should always be visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import seaborn as sns; sns.set_theme(style='white')
from scipy import stats
from scipy import signal
from sklearn.decomposition import PCA

if input('Regenerate data? (y/n) DEFAULT = n \n')=='y':
##### Load Data
    print('REGENERATING...\n')
    t_delay=10 #Takens delay
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
    # Prediction
    #############################################

    train_len=5000    # Number of datasteps used in training
    prediction_len=1000 #Timesteps predicted into the future after training
    discard=1000    #Wait for the Network to forget
    dim=1000  #Dimension of the Network

    u_data=data

    X=np.zeros((train_len+discard+1,dim)) 

    rng=np.random.default_rng(12345)
    X[0]=rng.random(dim) #Initialize X[0] randomly

    from Libs.Net import Network
    net=Network(dim,alpha=0.99,a=0.5)

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

    #Create the X training data (x_n-1, x_n)
    for i in range(delay):
        X_train[:,i*dim:(i+1)*dim]=X[discard-delay+i+1:discard-delay+i+1+train_len]

    ### Training Gamma. 
    from Libs.Train_Gamma import Train_NN_PCA
    trainer=Train_NN_PCA(X_train,Y_train,hidden_layers=12,layer_dimension=64,epochs=150,batch_size=256)


    #############################################
    # Prediction
    #############################################
    X_pred = np.zeros((prediction_len, 2*dim))

    # Setting up variables
    u_predicted=np.zeros((prediction_len,u_data.shape[1]))
    u=u_data[train_len+discard]
    x_0=X[train_len+discard]
    u_predicted[0]=u
    X_stack=np.zeros(delay*dim)
    noise_points = []
    for i in range(delay):
        X_stack[i*dim:(i+1)*dim]=X[train_len+discard-delay+i+1]
    S=train_len+discard
    rng=np.random.default_rng(12345)
    # Running Prediction
    for t in range(prediction_len-1):
        print("Predicted {} out of {} \r".format(t,prediction_len),end='')
        x_1=net.g(u,x_0)

        for i in range(delay-1):
            X_stack[i*dim:(i+1)*dim]=X_stack[(i+1)*dim:(i+2)*dim]

        X_stack[(delay-1)*dim:delay*dim]=x_1
        if t%100 == 0: #Add noise every 100 steps to show global dissipativity
            noise = rng.normal(0,2,X_stack.shape)
            noise_points.append(X_stack)
            X_stack += noise
        X_pred[t] = X_stack
        u_new=np.zeros(t_delay)
        for i in range(t_delay-1):
            u_new[i]=u[i+1]
            
        u_new[t_delay-1]=trainer.predict(X_stack)
        u_predicted[t+1]=u_new

        x_0=x_1
        u=u_new
    pca = PCA(3)

    
    actual_pca = pca.fit_transform(X_train)
    pred_pca = pca.transform(X_pred)
    
    fig=plt.figure(figsize =(8,4))
    axs0 =fig.subplots()


    axs0.plot(range(1000),u_predicted[0:1000,0],lw=0.5,color='darkblue')
    axs0.plot(range(1000),u_data[0:1000,0],lw=0.5,color='red')
    
    axs0.xaxis.set_tick_params(labelsize = 22)
    axs0.yaxis.set_tick_params(labelsize = 22)
    
    plt.savefig('Img/PCA/Trajectory.png')
    #############################################
    # Saving Data
    #############################################
    np.savetxt('Predicted_Data/PCA/PCA_predicted.txt',pred_pca)
    np.savetxt('Predicted_Data/PCA/PCA_actual.txt',actual_pca)
    
pred_pca = np.loadtxt('Predicted_Data/PCA/PCA_predicted.txt')    
actual_pca = np.loadtxt('Predicted_Data/PCA/PCA_actual.txt')
    
noise_pts = pred_pca[400::100]

plt.rcParams['figure.figsize'] = [8,4]
spec = gridspec.GridSpec( nrows=1,ncols=2,
                          wspace=0, width_ratios=[1, 1])
fig = plt.figure()
ax0 = fig.add_subplot(spec[1], projection='3d')
ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax0.zaxis.set_ticklabels([])
ax0.plot(pred_pca[400:,0], pred_pca[400:,1], pred_pca[400:,2], lw = 0.3, color = 'blue')
ax0.scatter(noise_pts[:,0], noise_pts[:,1], noise_pts[:,2])
ax1 = fig.add_subplot(spec[0], projection='3d')

ax1.plot(actual_pca[4400:,0], actual_pca[4400:,1], actual_pca[4400:,2], lw = 0.3, color = 'red')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.zaxis.set_ticklabels([])

plt.savefig('Img/PCA/Attractor.png')