
# Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import ruamel.yaml
import math
# For data modeling
import numpy as np
import tarfile
from matplotlib import cm
import warnings
import yaml
yaml_format = ruamel.yaml.YAML()
warnings.filterwarnings('ignore')
import os

# Get the directory of the current script or module
script_directory = os.path.dirname(os.path.realpath(__file__))

# Change the working directory to the script directory
os.chdir(script_directory)

# Now the working directory is the same as the directory of the script
print("Changed Working Directory:", os.getcwd())

cluster_num = 0

# =============================================================================
# Experiment selection and load data
# =============================================================================
# Getting the right paths
exp_type = 'identification' 
experiment_dir = './experiment_data/'
print(next(os.walk(experiment_dir)))
clusters = [item for item in next(os.walk(experiment_dir))[1] if exp_type in item]
if (exp_type == 'stairs') or (exp_type == 'static_characteristic'):
    experiment_type = 'identification'
else:
    experiment_type = exp_type

traces = {} 
traces_tmp = {}
for cluster in clusters:
    traces[cluster] = pd.DataFrame()
    # print("...",next(os.walk(experiment_dir+cluster))[2])
    if next(os.walk(experiment_dir+cluster))[1] == []:
        files = os.listdir(experiment_dir+cluster)          
        for fname in files:
            if fname.endswith("tar.xz"):
                tar = tarfile.open(experiment_dir+cluster+'/'+fname, "r:xz") 
                tar.extractall(path=experiment_dir+cluster+'/'+fname[:-7])
                tar.close()
    traces[cluster][0] = next(os.walk(experiment_dir+cluster))[1] 
# print("---------",next(os.walk(experiment_dir+cluster))[2])

# Processing data format to dataframe
data = {}
for cluster in clusters:
    data[cluster] = {}
    for trace in traces[cluster][0]:
        data[cluster][trace] = {}
        folder_path = experiment_dir+cluster+'/'+trace 
        # if os.path.isfile(folder_path+'/parameters.yaml'):
        #     with open(folder_path+"/parameters.yaml") as file:
        #         data[cluster][trace]['parameters'] = yaml.load(file, Loader=yaml.FullLoader)
        #         with open(folder_path+'/'+data[cluster][trace]['parameters']['config-file']) as file:
        #             data[cluster][trace]['parameters']['config-file'] = yaml.load(file, Loader=yaml.FullLoader)
        data[cluster][trace]['identification-runner-log'] = pd.read_csv(folder_path+"/"+experiment_type+"-runner.log", sep = '\0', names = ['created','levelname','process','funcName','message'])
        data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['identification-runner-log'][data[cluster][trace]['identification-runner-log']['funcName'] == 'enforce_powercap']
        data[cluster][trace]['enforce_powercap'] = data[cluster][trace]['enforce_powercap'].set_index('created')
        data[cluster][trace]['enforce_powercap']['powercap'] = [''.join(c for c in data[cluster][trace]['enforce_powercap']['message'][i] if c.isdigit()) for i in data[cluster][trace]['enforce_powercap'].index]
        # Loading sensors data files
        # print(cluster,trace)
        pubMeasurements = pd.read_csv(folder_path+"/dump_pubMeasurements.csv")
        pubProgress = pd.read_csv(folder_path+"/dump_pubProgress.csv")
         # Extracting sensor data
        rapl_sensor0 = rapl_sensor1 = rapl_sensor2 = rapl_sensor3 = downstream_sensor = pd.DataFrame({'timestamp':[],'value':[]})
        for i, row in pubMeasurements.iterrows():
            if row['sensor.id'] == 'RaplKey (PackageID 0)':
                rapl_sensor0 = rapl_sensor0.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 1)':
                rapl_sensor1 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 2)':
                rapl_sensor2 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
            elif row['sensor.id'] == 'RaplKey (PackageID 3)':
                rapl_sensor3 = rapl_sensor1.append({'timestamp':row['sensor.timestamp'],'value':row['sensor.value']}, ignore_index=True)
        progress_sensor = pd.DataFrame({'timestamp':pubProgress['msg.timestamp'],'value':pubProgress['sensor.value']})
        # Writing in data dict
        data[cluster][trace]['rapl_sensors'] = pd.DataFrame({'timestamp':rapl_sensor0['timestamp'],'value0':rapl_sensor0['value'],'value1':rapl_sensor1['value'],'value2':rapl_sensor2['value'],'value3':rapl_sensor3['value']})
        data[cluster][trace]['performance_sensors'] = pd.DataFrame({'timestamp':progress_sensor['timestamp'],'progress':progress_sensor['value']})
        #data[cluster][trace]['nrm_downstream_sensors'] = pd.DataFrame({'timestamp':downstream_sensor['timestamp'],'downstream':downstream_sensor['value']})
        # Indexing on elasped time since the first data point
        data[cluster][trace]['first_sensor_point'] = min(data[cluster][trace]['rapl_sensors']['timestamp'][0], data[cluster][trace]['performance_sensors']['timestamp'][0])#, data[cluster][trace]['nrm_downstream_sensors']['timestamp'][0])
        data[cluster][trace]['rapl_sensors']['elapsed_time'] = (data[cluster][trace]['rapl_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
        data[cluster][trace]['rapl_sensors'] = data[cluster][trace]['rapl_sensors'].set_index('elapsed_time')
        data[cluster][trace]['performance_sensors']['elapsed_time'] = (data[cluster][trace]['performance_sensors']['timestamp'] - data[cluster][trace]['first_sensor_point'])
        data[cluster][trace]['performance_sensors'] = data[cluster][trace]['performance_sensors'].set_index('elapsed_time')

# Compute extra metrics: averages, frequencies, upsampling
for cluster in clusters:
    for trace in traces[cluster][0]:
        # Average sensors value
        avg0 = data[cluster][trace]['rapl_sensors']['value0'].mean()
        avg1 = data[cluster][trace]['rapl_sensors']['value1'].mean()
        avg2 = data[cluster][trace]['rapl_sensors']['value2'].mean()
        avg3 = data[cluster][trace]['rapl_sensors']['value3'].mean()
        data[cluster][trace]['aggregated_values'] = {'rapl0':avg0,'rapl1':avg1,'rapl2':avg2,'rapl3':avg3,'progress':data[cluster][trace]['performance_sensors']['progress']}#'rapl0_std':std0,'rapl1_std':std1,'rapl2_std':std2,'rapl3_std':std3,'downstream':data[cluster][trace]['nrm_downstream_sensors']['downstream'].mean(),'progress':data[cluster][trace]['performance_sensors']['progress']}
        avgs = pd.DataFrame({'averages':[avg0, avg1, avg2, avg3]})
        data[cluster][trace]['aggregated_values']['rapls'] = avgs.mean()[0]
        # Sensors periods and frequencies
            # RAPL
        rapl_elapsed_time = data[cluster][trace]['rapl_sensors'].index
        data[cluster][trace]['aggregated_values']['rapls_periods'] = pd.DataFrame([rapl_elapsed_time[t]-rapl_elapsed_time[t-1] for t in range(1,len(rapl_elapsed_time))], index=[rapl_elapsed_time[t] for t in range(1,len(rapl_elapsed_time))], columns=['periods'])
            # Progress
        performance_elapsed_time = data[cluster][trace]['performance_sensors'].index
        #data[cluster][trace]['aggregated_values']['performance_frequency'] = pd.DataFrame([performance_elapsed_time[t]/(performance_elapsed_time[t]-performance_elapsed_time[t-1]) for t in range(1,len(performance_elapsed_time))], index=[performance_elapsed_time[t] for t in range(1,len(performance_elapsed_time))], columns=['frequency'])
        progress_data = data[cluster][trace]['performance_sensors'].loc[:,'progress'].values.copy()

        data[cluster][trace]['aggregated_values']['performance_frequency'] = pd.DataFrame([progress_data[t]/(performance_elapsed_time[t]-performance_elapsed_time[t-1]) for t in range(1,len(performance_elapsed_time))], index=[performance_elapsed_time[t] for t in range(1,len(performance_elapsed_time))], columns=['frequency'])
        # Execution time:
        data[cluster][trace]['aggregated_values']['execution_time'] = performance_elapsed_time[-1]
        data[cluster][trace]['aggregated_values']['upsampled_timestamps'] = data[cluster][trace]['rapl_sensors'].index
        # Computing count and frequency at upsampled_frequency:
        data[cluster][trace]['aggregated_values']['progress_frequency_median'] = pd.DataFrame({'median':np.nanmedian(data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where(data[cluster][trace]['aggregated_values']['performance_frequency'].index<= data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0],0)),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
        idx = 0  # index of powercap change in log
        data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame({'pcap':math.nan,'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
        for t in range(1,len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
             data[cluster][trace]['aggregated_values']['progress_frequency_median'] = data[cluster][trace]['aggregated_values']['progress_frequency_median'].append({'median':np.nanmedian(data[cluster][trace]['aggregated_values']['performance_frequency']['frequency'].where((data[cluster][trace]['aggregated_values']['performance_frequency'].index>= data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t-1]) & (data[cluster][trace]['aggregated_values']['performance_frequency'].index <=data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]))),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
             if (experiment_type == 'controller') or (experiment_type == 'identification'): 
                 if (float(data[cluster][trace]['enforce_powercap'].index[idx])-data[cluster][trace]['first_sensor_point'])<data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                     if idx < len(data[cluster][trace]['enforce_powercap'])-1:           
                         idx = idx +1
                 if (float(data[cluster][trace]['enforce_powercap'].index[0])-data[cluster][trace]['first_sensor_point'])>data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                    data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':math.nan,'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
                 elif (float(data[cluster][trace]['enforce_powercap'].index[-1])-data[cluster][trace]['first_sensor_point'])<data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]:
                     data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['enforce_powercap']['powercap'].iloc[-1]),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
                 else:
                     data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['enforce_powercap']['powercap'].iloc[idx-1]),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
        data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].set_index('timestamp')
        if (experiment_type == 'preliminaries') or (experiment_type == 'static_characteristic'):
            print("cheking execution of this line.")
            data[cluster][trace]['aggregated_values']['pcap'] = pd.DataFrame({'pcap':int(data[cluster][trace]['parameters']['powercap']),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][0]}, index=[0])
            for t in range(1,len(data[cluster][trace]['aggregated_values']['upsampled_timestamps'])):
                data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].append({'pcap':int(data[cluster][trace]['parameters']['powercap']),'timestamp':data[cluster][trace]['aggregated_values']['upsampled_timestamps'][t]}, ignore_index=True)
            data[cluster][trace]['aggregated_values']['pcap'] = data[cluster][trace]['aggregated_values']['pcap'].set_index('timestamp')
        data[cluster][trace]['aggregated_values']['progress_frequency_median'] = data[cluster][trace]['aggregated_values']['progress_frequency_median'].set_index('timestamp')


#print(f"________{data[cluster][trace]}__")
# =============================================================================
# STAIRS
# =============================================================================

# FIGURE 3
#print(f"The parameters in the content is {data[cluster][trace]['parameters']}")
cluster_num_t = 0
tot_cluster = len(clusters)
num_rows = int(tot_cluster ** 0.5)
num_cols = (tot_cluster + num_rows - 1) // num_rows
fig_P, axes_P = plt.subplots(nrows=num_rows, ncols=num_cols)
if tot_cluster > 1:
    axes_P = axes_P.ravel()
else:
    axes_P = [axes_P]
margin = 0.3937  
top_margin = 1 * margin / fig_P.get_figheight()
bottom_margin = 1 * margin / fig_P.get_figheight()
left_margin = 1 * margin / fig_P.get_figwidth()
right_margin = 1 * margin / fig_P.get_figwidth()

fig_P.subplots_adjust(
     top=1-top_margin,
     bottom=bottom_margin,
     left=left_margin,
     right=1-right_margin,
     hspace=0.25,
     wspace=0.2
)
# axes_P.set_ylabel('Measured Power [W]')
# axes_P.set_ylabel('Measured Power [W]')
# x_zoom = [0,150]
y_zoom = [0,150]
for k,cluster in enumerate(clusters):
    # axes_P.set_xlim(x_zoom)
    # axes_P.set_ylim(y_zoom)
    axes_P[k].grid(True)
    for my_trace in traces[cluster][0]:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5.7,6.6))
        data[cluster][my_trace]['aggregated_values']['progress_frequency_median']['median'].plot(color='k',ax=axes, marker='o', markersize=3,linestyle='')
        axes.set_ylabel('Progress [Hz]')
        axes.set_xlabel('')
        axes.legend(['Measure','Objective value','Objective value ±5%'],fontsize='small')
        # axes.set_xlim(x_zoom)
        axes.grid(True)
        data[cluster][my_trace]['aggregated_values']['pcap'].plot(color='k',ax=axes, style=".", label='Powercap')#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value0'].plot(color='lightcoral',ax=axes, marker="+", linestyle='',label = 'Measure')#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value1'].plot(color='lightcoral',ax=axes, marker="+", linestyle='')
        data[cluster][my_trace]['rapl_sensors']['value2'].plot(color='lightcoral',ax=axes, marker="+", linestyle='')#, style="+",  markersize=4)
        data[cluster][my_trace]['rapl_sensors']['value3'].plot(color='lightcoral',ax=axes, marker="+", linestyle='')#, style="+",  markersize=4)
        axes.set_ylabel('Power [W]')
        axes.set_xlabel('Time [s]')
        # axes.legend(['Powercap','Measure'],fontsize='small',ncol=1) # ,'Measure - package1'
        handles, labels = axes.get_legend_handles_labels()
        powercap_legend = plt.legend(handles[:1], labels[:1], fontsize='small', loc='upper left')
        measure_legend = plt.legend(handles[1:], labels[1:], fontsize='small', loc='upper right')
        axes.add_artist(powercap_legend)
        axes.grid(True)
        # axes.set_xlim(x_zoom)

        # ==========================
        # PLOT OF MEASURE POWER VS PCAP
        # ==========================

        # data[cluster][my_trace]['aggregated_values']['progress_frequency_median']['median'].plot(color='k',ax=axes[0], marker='o', markersize=3,linestyle='')

        # axes_P.legend([''],fontsize='small')
        X_DATA = data[cluster][my_trace]['aggregated_values']['pcap']
        X_AVG = X_DATA.mean(axis=0)
        Y_DATA = data[cluster][my_trace]['rapl_sensors'].mean(axis=0)
        Y_AVG = Y_DATA.iloc[1:].mean(axis=0)
        axes_P[k].plot(X_AVG,Y_AVG,color='red', marker="+", linestyle='', markersize = 2)
    title = f"{cluster}"
    axes_P[k].set_title(title,fontsize=5, color = 'blue')
    fig.savefig(f'./RESULTS/fig_3_{cluster}.pdf')
    axes_P[k].set_xlabel('PCAP')
    axes_P[k].tick_params(labelsize=4)
    axes_P[k].tick_params(labelsize=4)
    cluster_num_t += 1
fig_P.savefig(f'./RESULTS/PCAP_vs_P.pdf')



# df1 = pd.DataFrame(data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'])
# df2 = pd.DataFrame(data[cluster][trace]['aggregated_values']['pcap'])
# df3 = pd.DataFrame(data[cluster][trace]['rapl_sensors']['value0'])
# df4 = pd.DataFrame(data[cluster][trace]['rapl_sensors']['value1'])
# merged_df = pd.merge(df1, df2, on='timestamp')
# merged_df = pd.merge(merged_df, df3, left_index=True, right_index= True)
# merged_df = pd.merge(merged_df, df4, left_index=True, right_index= True)

# # merged_df = df1.merge(df2,left_on='timestamp',right_index=True).reset_index()
# # merged_df = merged_df.merge(df3,left_on='elapsed_time',right_index=True).reset_index()
# # merged_df = merged_df.merge(df4,left_on='elapsed_time',right_index=True).reset_index()
# merged_df.to_csv('merged_data.csv', index=True)


all_dataframes = []

for cluster in clusters:
    for trace in traces[cluster][0]:
        df1 = pd.DataFrame(data[cluster][trace]['aggregated_values']['progress_frequency_median']['median'])
        df2 = pd.DataFrame(data[cluster][trace]['aggregated_values']['pcap'])
        df3 = pd.DataFrame(data[cluster][trace]['rapl_sensors']['value0'])
        df4 = pd.DataFrame(data[cluster][trace]['rapl_sensors']['value1'])
        
        # Merge dataframes
        merged_df = pd.merge(df1, df2, on='timestamp')
        merged_df = pd.merge(merged_df, df3, left_index=True, right_index=True)
        merged_df = pd.merge(merged_df, df4, left_index=True, right_index=True)
        merged_df['cluster'] = cluster     
        # Add merged dataframe to the list
        all_dataframes.append(merged_df)

# Concatenate all dataframes in the list along the rows
merged_data = pd.concat(all_dataframes)

# Save merged dataframe to CSV file
merged_data.to_csv('merged_data.csv', index=True)

