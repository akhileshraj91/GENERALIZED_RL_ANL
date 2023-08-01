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
import data_generator as RLDG
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
OUTPUT_DIR = './RESULTS/'
yaml_format = ruamel.yaml.YAML()
PARAMS_PATH = experiment_dir+"PARAMS/"
param_files = os.listdir(PARAMS_PATH)
for file in param_files:
    if 'params' in file:
        und_index = file.find('-identification')
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

# print(APPLICATIONS)


# print(clusters)
num_subplots = len(APPLICATIONS)
num_rows = int(num_subplots ** 0.5)
num_cols = (num_subplots + num_rows - 1) // num_rows

colors = sns.color_palette("colorblind", num_subplots)

fig_pareto, axes_pareto = plt.subplots(nrows=num_rows, ncols=num_cols)
axes_pareto = [axes_pareto]
margin = 0.3937  
top_margin = 1 * margin / fig_pareto.get_figheight()
bottom_margin = 1 * margin / fig_pareto.get_figheight()
left_margin = 1 * margin / fig_pareto.get_figwidth()
right_margin = 1 * margin / fig_pareto.get_figwidth()

fig_pareto.subplots_adjust(
     top=1-top_margin,
     bottom=bottom_margin,
     left=left_margin,
     right=1-right_margin,
     hspace=0.25,
     wspace=0.2
)


EX_DIRS = [item for item in next(os.walk('./experiment-data/'))[1] if 'control' in item]
print(EX_DIRS)
for ex in EX_DIRS:
     experiment_dir = f'./experiment-data/{ex}/'
     # print(experiment_dir)
     clusters = next(os.walk(experiment_dir))[1]
     # print(clusters)
     for cluster in clusters:
          print(cluster)
          print(APPLICATIONS)
          k = APPLICATIONS.index(cluster)
          data,traces = RLDG.generate_data(experiment_dir,cluster)
          pareto = {}

          for trace in traces[cluster][0]:
               data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,4)])
               if 'RL' in ex:
                    nof = data[cluster][trace]['weights'][0]
                    ind1 = nof.find('-')
                    ind2 = nof.find('---')
                    c_1 = round(float(nof[ind1+1:ind2]),1)
                    c_2 = round(float(nof[ind2+3:]),1)
                    data[cluster][trace]['label'] = (c_1,c_2)
                    col = 'r'
               else:
                    data[cluster][trace]['label'] = None
                    col = 'k'
               # print(data[cluster][trace]['label'])
          pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1] for trace in traces[cluster][0]],'Labels': [data[cluster][trace]['label'] for trace in traces[cluster][0]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3 for trace in traces[cluster][0]])
          pareto[cluster].sort_index(inplace=True)

          execution_time_power = pareto[cluster].index*pareto[cluster]['Execution Time']
     
          cmap = cm.get_cmap('viridis')
          cb = axes_pareto[k].scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.', color=col, s=30, label=f'{ex}')
          x_data = pareto[cluster].index
          y_data = pareto[cluster]['Execution Time']
          z_data = pareto[cluster]['Labels'].iloc()
          EDP_values = x_data*y_data

          for i,data in enumerate(zip(x_data,y_data)):
               axes_pareto[k].text(data[0]+0.03,data[1]+0.03,z_data[i],fontsize=2)

          axes_pareto[k].grid(True)
          axes_pareto[k].set_ylabel('Execution time [s]',fontsize = 3.5)
          axes_pareto[k].set_xlabel('Energy consumption [kJ]', fontsize = 3.5)
          axes_pareto[k].tick_params(axis='x', labelsize=4)
          axes_pareto[k].tick_params(axis='y', labelsize=4)
          axes_pareto[k].legend(fontsize = 3)
          title = f"{cluster}"
          axes_pareto[k].set_title(title, fontsize=5, color = 'blue')
     ##########################################################################################################################






     fig_pareto.savefig(OUTPUT_DIR+"result_"+str(now)+".pdf")
     # plt.savefig("./figures_normal/result_"+str(now)+".png")


     plt.show()

