import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import math
# For data modeling
import scipy.optimize as opt
import numpy as np
import tarfile
from matplotlib import cm
import seaborn as sns
import pickle
import ruamel.yaml

# from stable_baselines3 import PPO
from stable_baselines3 import PPO

# import new_code_normalized_Action as ncna
from datetime import datetime
#
#fig, axs = plt.subplots(2)
#fig.suptitle('power and performance against time')
now = datetime.now()

a = {}
b = {}
alpha = {}
beta = {}
K_L = {}
APPLICATIONS = []

experiment_dir = './'
yaml_format = ruamel.yaml.YAML()
PARAMS_PATH = experiment_dir+"PARAMS_backup/"
param_files = os.listdir(PARAMS_PATH)
for file in param_files:
    if 'params' in file:
        und_index = file.find('_')
        name = file[0:und_index]
        APPLICATIONS.append(name)
        with open(PARAMS_PATH+file) as files:
            parameters = yaml_format.load(files)
            a[name] = parameters['rapl']['slope']
            b[name] = parameters['rapl']['offset']
            alpha[name] = parameters['model']['alpha']
            beta[name] = parameters['model']['beta']
            K_L[name] = parameters['model']['gain']
            files.close()

print(APPLICATIONS)

#
# cluster = 'EP_GN_RL_control'
# file1 = open(r'data_dir'+str(cluster),'rb')
# file2 = open(r'trace_dir'+str(cluster),'rb')#
# data = pickle.load(file1)
# traces = pickle.load(file2)
# #
# #
# #
# pareto = {}
# #
# for trace in traces[cluster][0]:
#      data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,4)])
#      nof = data[cluster][trace]['weights'][0]
#      ind1 = nof.find('_')
#      ind2 = nof.find('___')
#      c_1 = round(float(nof[ind1+1:ind2]),1)
#      c_2 = round(float(nof[ind2+3:]),1)
#      # print(c_1,c_2)
#      data[cluster][trace]['label'] = (c_1,c_2)
#      print(data[cluster][trace]['label'])
# pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1] for trace in traces[cluster][0]],'Labels': [data[cluster][trace]['label'] for trace in traces[cluster][0]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3 for trace in traces[cluster][0]])
# pareto[cluster].sort_index(inplace=True)
# #
# #
# execution_time_power = pareto[cluster].index*pareto[cluster]['Execution Time']
# #
# #
# # # FIGURE 7
# cmap = cm.get_cmap('viridis')
# #
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
# cb = axes.scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.', c='r', s=30, label='RL-Controller on EP')

# x_data = pareto[cluster].index
# y_data = pareto[cluster]['Execution Time']
# z_data = pareto[cluster]['Labels'].iloc()
# print(x_data)
# print(y_data)
# print(z_data)
# # print(len(x_data),len(y_data),len(z_data))

# # fig2,axes2 = plt.subplots(nrows=1,ncols=1,figsize=(6.6,6.6))
# for i,data in enumerate(zip(x_data,y_data)):
#      # axes.plot(data[0],data[1])
#      axes.text(data[0]+0.03,data[1]+0.03,z_data[i],fontsize=2)

# # fig2.savefig("./figures_normal/result_"+str(now)+".pdf")

# # axes.grid(True)
# # axes.set_ylabel('Execution time [s]')
# # axes.set_xlabel('Energy consumption [kJ]')



##########################################################################################################################
cluster = 'GN_RL_controller'
file1 = open(r'data_dir'+str(cluster),'rb')
file2 = open(r'trace_dir'+str(cluster),'rb')#
data = pickle.load(file1)
traces = pickle.load(file2)
#
#
#
pareto = {}
#
for trace in traces[cluster][0]:
     data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,4)])
     nof = data[cluster][trace]['weights'][0]
     ind1 = nof.find('_')
     ind2 = nof.find('___')
     c_1 = round(float(nof[ind1+1:ind2]),1)
     c_2 = round(float(nof[ind2+3:]),1)
     # print(c_1,c_2)
     data[cluster][trace]['label'] = (c_1,c_2)
     print(data[cluster][trace]['label'])
pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1] for trace in traces[cluster][0]],'Labels': [data[cluster][trace]['label'] for trace in traces[cluster][0]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3 for trace in traces[cluster][0]])
pareto[cluster].sort_index(inplace=True)
#
#
execution_time_power = pareto[cluster].index*pareto[cluster]['Execution Time']
#
#
# # FIGURE 7
cmap = cm.get_cmap('viridis')
#
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
cb = axes.scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.', c='k', s=30, label='RL-Controller on STREAM')
#
x_data = pareto[cluster].index
y_data = pareto[cluster]['Execution Time']
z_data = pareto[cluster]['Labels'].iloc()
print(x_data)
print(y_data)
print(z_data)
# print(len(x_data),len(y_data),len(z_data))

# fig2,axes2 = plt.subplots(nrows=1,ncols=1,figsize=(6.6,6.6))
for i,data in enumerate(zip(x_data,y_data)):
     # axes.plot(data[0],data[1])
     axes.text(data[0]+0.03,data[1]+0.03,z_data[i],fontsize=2)

axes.grid(True)
axes.set_ylabel('Execution time [s]')
axes.set_xlabel('Energy consumption [kJ]')
axes.legend()
title = "Comparing RL and PI controller with varying reward fucntions and varying setpoints."
axes.set_title(title,fontsize=8)
##########################################################################################################################






fig.savefig("./figures_normal/result_"+str(now)+".pdf")
# plt.savefig("./figures_normal/result_"+str(now)+".png")


plt.show()
