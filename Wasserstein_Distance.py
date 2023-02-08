"""
NOTE: This script requires the scipy library

This script uses data from the folder {Predicted_Data} 

This script generates the table for the supplementary document: Table 1
"""

from scipy import stats
import numpy as np
import sys


Ns=[1000,2000,5000,10000]

f = open("Wasserstein_Distances.txt", "w")

f.write("===============================\n")
f.write("======= Double Pendulum =======\n")
f.write("===============================\n")

f.write('============= ESN =============\n')
u_predicted=np.loadtxt('Predicted_Data/DP/esn_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/DP/esn_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")

f.write('============= RCN =============\n')
u_predicted=np.loadtxt('Predicted_Data/DP/rcn_predicted.txt')
u_data=np.loadtxt('Predicted_Data/DP/rcn_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")

f.write('============ Takens ===========\n')
u_predicted=np.loadtxt('Predicted_Data/DP/takens_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/DP/takens_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('============ Sindy ===========\n')
u_predicted=np.loadtxt('Predicted_Data/DP/sindy_predicted.txt')
u_data=np.loadtxt('Predicted_Data/DP/sindy_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")   
    
f.write("=============================\n")
f.write("======= Lorenz Course =======\n")
f.write("=============================\n")

f.write('============ ESN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Lorenz/esn_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Lorenz/esn_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('============ RCN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Lorenz/delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Lorenz/delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    

f.write('=========== Takens ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Lorenz/takens_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Lorenz/takens_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('=========== Sindy ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Lorenz/sindy_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Lorenz/sindy_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
    
f.write("=============================\n")
f.write("============ Henon ==========\n")
f.write("=============================\n")

f.write('============ ESN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Henon/esn_delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon/esn_delay_comp_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('============ RCN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Henon/delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon/delay_comp_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")

f.write('=========== Takens ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Henon/takens_delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon/takens_delay_comp_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('=========== Sindy ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Henon/henon_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon/henon_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
    
f.write("=============================\n")
f.write("========= Henon Fine ========\n")
f.write("=============================\n")

f.write('============ ESN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Henon_Fine/esn_delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon_Fine/esn_delay_comp_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('============ RCN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Henon_Fine/delay_fine_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon_Fine/delay_fine_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")

f.write('=========== Takens ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Henon_Fine/takens_delay_comp_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon_Fine/takens_delay_comp_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
f.write('=========== Sindy ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Henon_Fine/Sindy_Fine_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Henon_Fine/Sindy_Fine_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n,0],u_predicted[0:n,0])}\n")
    
    
f.write("=============================\n")
f.write("========= Logistic ==========\n")
f.write("=============================\n")

f.write('============ ESN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Logistic/esn_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Logistic/esn_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")

f.write('============ RCN ============\n')
u_predicted=np.loadtxt('Predicted_Data/Logistic/delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Logistic/delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")


f.write('=========== Sindy ==========\n')
u_predicted=np.loadtxt('Predicted_Data/Logistic/sindy_predicted.txt')
u_data=np.loadtxt('Predicted_Data/Logistic/sindy_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")

f.write("=============================\n")
f.write("============= PM ============\n")
f.write("=============================\n")

f.write('============ ESN ============\n')
u_predicted=np.loadtxt('Predicted_Data/PM/esn_delay_predicted.txt')
u_data=np.loadtxt('Predicted_Data/PM/esn_delay_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")
    
f.write('============ RCN ============\n')
u_predicted=np.loadtxt('Predicted_Data/PM/simple_1d_predicted.txt')
u_data=np.loadtxt('Predicted_Data/PM/simple_1d_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")

f.write('=========== Sindy ==========\n')
u_predicted=np.loadtxt('Predicted_Data/PM/sindy_predicted.txt')
u_data=np.loadtxt('Predicted_Data/PM/sindy_actual.txt')
for n in Ns:
    f.write(f"Wasserstein distance for {n} timesteps: {stats.wasserstein_distance(u_data[0:n],u_predicted[0:n])}\n")
    
f.close()
