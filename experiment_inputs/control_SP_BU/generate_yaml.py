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

experiment_dir = '/home/cc/ANL_comparison/'

with open(experiment_dir+'experiment_data/params.yaml') as files:
    parameters = yaml.load(files)
    print(parameters)

for cfg in next(os.walk(experiment_dir+'experiment_inputs/control_SP/'))[2]:
    print(cfg)
    if ".yaml" in cfg:
        print(f"_________________{cfg}")
        hyp_ind = cfg.find('-')
        dot_ind = cfg.find('.')
        und_ind = cfg.find('_')
        with open(cfg,'w') as fil:
            #new_params = yaml.load(fil)
            sp = cfg[hyp_ind+1:dot_ind]
            parameters['controller']['setpoint'] = float(sp)/100
            print(parameters)
            yaml.dump(parameters,fil)
#            os.rename(cfg, 'CC_'+cfg[und_ind+1:])

