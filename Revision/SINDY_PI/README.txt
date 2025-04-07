README

Notes for running code:


1. Please make sure everything is in the same folder, and make sure your matlab has the path to this folder.

2. dp.mat is the raw data from python. d_dp.mat is the derivate that I got from using the function in DP_generator.py.   We needed some additional initial conditions to test it on, I used the following values
[theta1 = pi/5, theta2 = 0.1, omega1 = -1.2, omega2 = 0.5], and we got dp2.mat, and d_dp2.mat (the derivative). 


3.a) To be able to change the initial values and conditions, please add the following to the end of DP_generator.py

## Extra code for derivatives
varss = [data['theta1'], data['theta2'], data['omega1'], data['omega2']]
d_data = DoublePendulumDerivArray(varss, data['times'], l1=1.15,l2=1,m1=0.75,m2=2.5,g=9.81)
df = pd.DataFrame({'theta1': data['theta1'], 'theta2': data['theta2'], 'omega1':data['omega1'], 'omega2':data['omega2'], 't': data['times']})
scipy.io.savemat('dp.mat', {'struct':df.to_dict("list")})

d_df = pd.DataFrame({'d_theta1': d_data[0,:], 'd_theta2':d_data[1,:], 'd_omega1':d_data[2,:], 'd_omega2':d_data[3,:]})
scipy.io.savemat('d_dp.mat', {'d_struct':d_df.to_dict("list")})


data2 = DoublePendulumTrajectory(initial_theta1=numpy.pi/5,initial_theta2=0.1,initial_omega1=-1.2,initial_omega2=0.5,
           timestep=0.05,numstep=10000,l1=1+1*.15,l2=1,m1=0.25*3,m2=2.5,g=9.81)
varss2 = [data2['theta1'], data2['theta2'], data2['omega1'], data2['omega2']]
d_data2 = DoublePendulumDerivArray(varss2, data2['times'], l1=1.15,l2=1,m1=0.75,m2=2.5,g=9.81)
df2 = pd.DataFrame({'theta1': data2['theta1'], 'theta2': data2['theta2'], 'omega1':data2['omega1'], 'omega2':data2['omega2'], 't': data2['times']})
scipy.io.savemat('dp2.mat', {'struct2':df2.to_dict("list")})

d_df2 = pd.DataFrame({'d_theta1': d_data2[0,:], 'd_theta2':d_data2[1,:], 'd_omega1':d_data2[2,:], 'd_omega2':d_data2[3,:]})
scipy.io.savemat('d_dp2.mat', {'d_struct2':d_df2.to_dict("list")})



3.b) The code above will do what was mentioned in 2, and save the files as .mat files. Please just make sure they save in the same place as where you have the folder. 


4. predicted.mat, is the name of the predicted values. This corresponds to dp.mat.  

5. You only need to run MAIN_FILE_TO_RUN.m

6. The output in the command window will give you the system of ODEs, I suggest scrolling a little bit up, where it shows you each best one individually. Some trigonometric manipulation will be needed, if you want to compare it term to term with the original system. 




