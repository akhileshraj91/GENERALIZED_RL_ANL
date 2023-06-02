import os
import pandas as pd
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
#import yaml
import math
# For data modeling
import scipy.optimize as opt
import numpy as np
import tarfile
yaml = YAML()

experiment_dir = './PARAMS_backup/'
parameter_files = next(os.walk(experiment_dir))[2]
print(parameter_files)

OUTPUT_DIR = './experiment_inputs/control_SP/'
SP_files = next(os.walk(OUTPUT_DIR))[1]
print(SP_files)

# for cfg in next(os.walk(experiment_dir+'experiment_inputs/control_SP/'))[2]:
for param in parameter_files:
    with open(experiment_dir+param) as FIL:
        parameters = yaml.load(FIL)  
    # print(f"_________________{param}")
    UND = param.find('_')
    SPF_name = param[:UND]
    print(SPF_name) 
    for file in next(os.walk(OUTPUT_DIR+SPF_name))[2]:
        if ".yaml" in file:
            print(file)
            hyp_ind = file.find('-')
            dot_ind = file.find('.')
            und_ind = file.find('_')
            with open(OUTPUT_DIR+SPF_name+f'/{file}','w') as fil:
                sp = file[hyp_ind+1:dot_ind]
                parameters['controller']['setpoint'] = float(sp)/100
                print(parameters)
                yaml.dump(parameters,fil)

