#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:27:26 2020

@author: sophiecerf
"""

# Libraries
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


exp_type = 'controller'  # ex: 'stairs' 'identification' 'static_characteristic' 'controller' XXX
experiment_dir = './experiment_data/'
clusters = next(os.walk(experiment_dir))[1]  # clusters are name of folders
print(clusters)
experiment_type = 'control'
# cluster = 'CC_RL_control'
traces = {}
traces_tmp = {}
for cluster in clusters:
    if experiment_type in cluster:
        print(cluster)
        traces[cluster] = pd.DataFrame()
        if next(os.walk(experiment_dir + cluster))[1] == []:
            files = os.listdir(experiment_dir + cluster)
            for fname in files:
                if fname.endswith("tar.xz"):
                    tar = tarfile.open(experiment_dir + cluster + '/' + fname, "r:xz")
                    tar.extractall(path=experiment_dir + cluster + '/' + fname[:-7])
                    tar.close()
        traces[cluster][0] = next(os.walk(experiment_dir + cluster))[1]

        traces[cluster][0] = next(os.walk(experiment_dir + cluster))[1]

        data = {}
        data[cluster] = {}

        for trace in traces[cluster][0]:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",trace)
            data[cluster][trace] = {}
            folder_path = experiment_dir + cluster + '/' + trace
            # Trace experimental plan: parameters or log
            if os.path.isfile(folder_path + '/SUCCESS'):
                data[cluster][trace]['SUCCESS'] = True
            else:
                data[cluster][trace]['SUCCESS'] = False
            if os.path.isfile(folder_path + '/parameters.yaml'):
                with open(folder_path + "/parameters.yaml") as file:
                    data[cluster][trace]['parameters'] = yaml.load(file, Loader=yaml.FullLoader)
                    # with open(folder_path + '/' + data[cluster][trace]['parameters']['config-file']) as file:
                        # data[cluster][trace]['parameters']['config-file'] = yaml.load(file, Loader=yaml.FullLoader)
            data[cluster][trace]['identification-runner-log'] = pd.read_csv(
                folder_path + "/" + experiment_type + "-runner.log", sep='\0',
                names=['created', 'levelname', 'process', 'funcName', 'message'])
            data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['identification-runner-log'][
                data[cluster][trace]['identification-runner-log']['funcName'] == 'enforce_powercap']
            data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['enforce_powercap'].set_index('created')
            data[cluster][trace]['enforce_powercap']['powercap'] = [
                ''.join(c for c in data[cluster][trace]['enforce_powercap']['message'][i] if c.isdigit()) for i in
                data[cluster][trace]['enforce_powercap'].index]
            all_files = os.listdir(folder_path)
            data[cluster][trace]['weights'] = [file for file in all_files if os.path.isfile(os.path.join(folder_path, file)) and file.startswith('dynamics')]
            print(">>>>>>>>>>>>>>>",data[cluster][trace]['weights'])
            # Loading sensors data files
            pubMeasurements = pd.read_csv(folder_path + "/dump_pubMeasurements.csv")
            pubProgress = pd.read_csv(folder_path + "/dump_pubProgress.csv")
            # Extracting sensor data
            rapl_sensor0 = rapl_sensor1 = rapl_sensor2 = rapl_sensor3 = downstream_sensor = pd.DataFrame(
                {'timestamp': [], 'value': []})
            for i, row in pubMeasurements.iterrows():
                if row['sensor.id'] == 'RaplKey (PackageID 0)':
                    rapl_sensor0 = rapl_sensor0.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},
                                                       ignore_index=True)
                elif row['sensor.id'] == 'RaplKey (PackageID 1)':
                    rapl_sensor1 = rapl_sensor1.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},
                                                       ignore_index=True)
                elif row['sensor.id'] == 'RaplKey (PackageID 2)':
                    rapl_sensor2 = rapl_sensor1.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},
                                                       ignore_index=True)
                elif row['sensor.id'] == 'RaplKey (PackageID 3)':
                    rapl_sensor3 = rapl_sensor1.append({'timestamp': row['sensor.timestamp'], 'value': row['sensor.value']},
                                                       ignore_index=True)
            progress_sensor = pd.DataFrame(
                {'timestamp': pubProgress['msg.timestamp'], 'value': pubProgress['sensor.value']})
            # Writing in data dict

            data[cluster][trace]['rapl_sensors'] = pd.DataFrame(
                {'timestamp': rapl_sensor0['timestamp'], 'value0': rapl_sensor0['value'], 'value1': rapl_sensor1['value'],
                 'value2': rapl_sensor2['value'], 'value3': rapl_sensor3['value']})
            data[cluster][trace]['performance_sensors'] = pd.DataFrame(
                {'timestamp': progress_sensor['timestamp'], 'progress': progress_sensor['value']})
            # data[cluster][trace]['nrm_downstream_sensors'] = pd.DataFrame({'timestamp':downstream_sensor['timestamp'],'downstream':downstream_sensor['value']})
            # Indexing on elasped time since the first data point
            data[cluster][trace]['first_sensor_point'] = min(data[cluster][trace]['rapl_sensors']['timestamp'][0],
                                                             data[cluster][trace]['performance_sensors']['timestamp'][
                                                                 0])  # , data[cluster][trace]['nrm_downstream_sensors']['timestamp'][0])
            data[cluster][trace]['rapl_sensors']['elapsed_time'] = (
                        data[cluster][trace]['rapl_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
            data[cluster][trace]['rapl_sensors'] = data[cluster][trace]['rapl_sensors'].set_index('elapsed_time')
            data[cluster][trace]['performance_sensors']['elapsed_time'] = (
                        data[cluster][trace]['performance_sensors']['timestamp'] - data[cluster][trace][
                    'first_sensor_point'])
            data[cluster][trace]['performance_sensors'] = data[cluster][trace]['performance_sensors'].set_index(
                'elapsed_time')


        for trace in traces[cluster][0]:
            # Average sensors value
            avg0 = data[cluster][trace]['rapl_sensors']['value0'].mean()
            avg1 = data[cluster][trace]['rapl_sensors']['value1'].mean()
            avg2 = data[cluster][trace]['rapl_sensors']['value2'].mean()
            avg3 = data[cluster][trace]['rapl_sensors']['value3'].mean()
            data[cluster][trace]['aggregated_values'] = {'rapl0': avg0, 'rapl1': avg1, 'rapl2': avg2, 'rapl3': avg3,
                                                         'progress': data[cluster][trace]['performance_sensors'][
                                                             'progress']}  # 'rapl0_std':std0,'rapl1_std':std1,'rapl2_std':std2,'rapl3_std':std3,'downstream':data[cluster][trace]['nrm_downstream_sensors']['downstream'].mean(),'progress':data[cluster][trace]['performance_sensors']['progress']}
            avgs = pd.DataFrame({'averages': [avg0, avg1, avg2, avg3]})
            data[cluster][trace]['aggregated_values']['rapls'] = avgs.mean()[0]
            # Sensors periods and frequencies
            # RAPL
            rapl_elapsed_time = data[cluster][trace]['rapl_sensors'].index
            data[cluster][trace]['aggregated_values']['rapls_periods'] = pd.DataFrame(
                [rapl_elapsed_time[t] - rapl_elapsed_time[t - 1] for t in range(1, len(rapl_elapsed_time))],
                index=[rapl_elapsed_time[t] for t in range(1, len(rapl_elapsed_time))], columns=['periods'])
            # Progress
            Progress_data = data[cluster][trace]['performance_sensors'].loc[:,'progress'].values.copy()
            performance_elapsed_time = data[cluster][trace]['performance_sensors'].index
            data[cluster][trace]['aggregated_values']['performance_frequency'] = pd.DataFrame(
                [Progress_data[t] / (performance_elapsed_time[t] - performance_elapsed_time[t - 1]) for t in
                 range(1, len(performance_elapsed_time))],
                index=[performance_elapsed_time[t] for t in range(1, len(performance_elapsed_time))], columns=['frequency'])
            # Execution time:
            data[cluster][trace]['aggregated_values']['execution_time'] = performance_elapsed_time[-1]
            data[cluster][trace]['aggregated_values']['upsampled_timestamps'] = data[cluster][trace]['rapl_sensors'].index
            # Computing count and frequency at upsampled_frequency:
            data[cluster][trace]['aggregated_values']['progress_frequency_median'] = pd.DataFrame({'median': np.nanmedian(
                data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where(
                    data[cluster][trace]['aggregated_values']['performance_frequency'].index <=
                    data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0], 0)), 'timestamp':
                                                                                                       data[cluster][trace][
                                                                                                           'aggregated_values'][
                                                                                                           'upsampled_timestamps'][
                                                                                                           0]}, index=[0])
            idx = 0  # index of powercap change in log
            data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame(
                {'pcap': math.nan, 'timestamp': data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]},
                index=[0])
            for t in range(1, len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
                data[cluster][trace]['aggregated_values']['progress_frequency_median'] = \
                data[cluster][trace]['aggregated_values']['progress_frequency_median'].append({'median': np.nanmedian(
                    data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where((data[cluster][trace]['aggregated_values'][
                                                                                                               'performance_frequency'].index >=
                                                                                                           data[cluster][
                                                                                                               trace][
                                                                                                               'aggregated_values'][
                                                                                                               'upsampled_timestamps'][
                                                                                                               t - 1]) & (
                                                                                                                      data[
                                                                                                                          cluster][
                                                                                                                          trace][
                                                                                                                          'aggregated_values'][
                                                                                                                          'performance_frequency'].index <=
                                                                                                                      data[
                                                                                                                          cluster][
                                                                                                                          trace][
                                                                                                                          'aggregated_values'][
                                                                                                                          'upsampled_timestamps'][
                                                                                                                          t]))),
                                                                                               'timestamp':
                                                                                                   data[cluster][trace][
                                                                                                       'aggregated_values'][
                                                                                                       'upsampled_timestamps'][
                                                                                                       t]},
                                                                                              ignore_index=True)




        # Parameters found using static Characteristic
        a = {'gros': 0.83, 'dahu': 0.94, 'yeti': 0.89}
        b = {'gros': 7.07, 'dahu': 0.17, 'yeti': 2.91}
        alpha = {'gros': 0.047, 'dahu': 0.032, 'yeti': 0.023}
        beta = {'gros': 28.5, 'dahu': 34.8, 'yeti': 33.7}
        K_L = {'gros': 25.6, 'dahu': 42.4, 'yeti': 78.5}
        # analytically found parameter
        tau = 0.33

        pareto = {}

        for trace in traces[cluster][0]:
            data[cluster][trace]['aggregated_values']['energy'] = np.nansum([np.nansum([data[cluster][trace]['rapl_sensors']['value'+str(package_number)].iloc[i+1]*data[cluster][trace]['aggregated_values']['rapls_periods'].iloc[i][0] for i in range(0,len(data[cluster][trace]['aggregated_values']['rapls_periods'].index))]) for package_number in range(0,4)])
        pareto[cluster] = pd.DataFrame({'Execution Time':[data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'].index[-1] for trace in traces[cluster][0]]},index=[data[cluster][trace]['aggregated_values']['energy']/10**3 for trace in traces[cluster][0]])
        pareto[cluster].sort_index(inplace=True)

        # FIGURE 7
        cmap = cm.get_cmap('viridis')

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.6,6.6))
        cb = axes.scatter(pareto[cluster].index,pareto[cluster]['Execution Time'], marker='.',s=30)
        #plt.show()
        plt.colorbar(cb,label='Degradation $\epsilon$ [unitless]')
        axes.grid(True)
        axes.set_ylabel('Execution time [s]')
        axes.set_xlabel('Energy consumption [kJ]')

        afile = open(r'data_dir'+str(cluster),'wb')
        bfile = open(r'trace_dir'+str(cluster),'wb')
        pickle.dump(data, afile)
        pickle.dump(traces, bfile)
        afile.close()
        bfile.close()
        # file2 = open(r'experiment_dir','rb')
        # new_d = pickle.load(file2)

        plt.show()
