from gym import Env
from gym.spaces import Box
from gym.spaces import MultiDiscrete
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import math
import ruamel.yaml
import os


a = {}
b = {}
alpha = {}
beta = {}
K_L = {}
APPLICATIONS = []

experiment_dir = './'
yaml_format = ruamel.yaml.YAML()
PARAMS_PATH = experiment_dir+"PARAMS/"
param_files = os.listdir(PARAMS_PATH)
for file in param_files:
    print(file)
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

T_S = 1
exec_steps = 10000                                                                                                      # Total clock cycles needed for the execution of program.
ACTION_MIN = 40                                                                                                         # Minima of control space (power cap), Please do not change while using mathematical model for simulations.
ACTION_MAX = 160                                                                                                        # Maxima of control space
ACT_MID = ACTION_MIN + (ACTION_MAX - ACTION_MIN) / 2                                                                    # Midpoint of the control space to compute the normalized action space
OBS_MAX = 200                                                                                                           # Maxima of observation space (performance)
OBS_MIN = 0                                                                                                             # Minima of observation space
OBS_MID = OBS_MIN + (OBS_MAX - OBS_MIN) / 2

exec_time = 10000
tau = 0.33

# print(APPLICATIONS[0])
def progress_funct(state, p_cap):                                                                                       # Function definition of the mathematical model of cluster performance and pcap relation.
    # p_now = abnormal_obs(state[0])
    print("The state transmitted are:",state)
    p_now = state[0]
    cluster = APPLICATIONS[state[1]]
    print(cluster)
    pcap_old_L = -np.exp(-alpha[cluster] * (a[cluster] * p_cap + b[cluster] - beta[cluster]))                           # Calculation of the PCAP for fitting it into the model.
    progress_value = K_L[cluster] * T_S / (T_S + tau) * pcap_old_L + tau / (T_S + tau) * (p_now - K_L[cluster]) + \
                     K_L[cluster]                                                                                       # Mathematical relation
    progress_NL = K_L[cluster] * (1 + pcap_old_L)
    # print("The next progress value is: ", progress_value)
    return_list = [round(progress_value[0]),state[1]]
    # print(return_list)
    # print(np.array(return_list))
    # return progress_value, progress_NL
    return return_list,progress_NL

def normal_obs(o):
    return (o - OBS_MIN) / (OBS_MAX - OBS_MIN)


def abnormal_obs(z):
    return z * (OBS_MAX - OBS_MIN) + OBS_MIN

def abnormal_action(a):
    return a * (ACTION_MAX-ACTION_MIN)/2 + ACT_MID

class Dynamical_Sys(Env):
    def __init__(self, exec_time, c_0=0, c_1=0):
        self.action_space = Box(low=-1, high=1, shape=(1,))
        self.observation_space = MultiDiscrete([160,2])
        self.execution_time = exec_time
        self.c_0 = c_0
        self.c_1 = c_1
        self.current_step = 0
        self.total_power = 0
        self.action = None
        self.state = None

    def step(self, action):
        # actual_state = abnormal_obs(self.state)
        actual_state = self.state
        actual_action = abnormal_action(action)
        # print("action is:", action)
        # print("action converted is:", actual_action)
        new_state, add_on = progress_funct(actual_state, actual_action)
        # normalized_new_state = normal_obs(new_state[0])
        self.state = np.array(new_state)
        # self.action = action[0]
        self.action = actual_action
        if new_state[0] > 0:
            self.current_step += new_state[0]
            reward_0 = -self.c_0 * self.action
            reward_1 = self.c_1 * self.state[0] / self.action
            reward = (reward_0 + reward_1)[0]

        else:
            reward = -100

        if self.current_step >= self.execution_time:
            done = True
        else:
            done = False
        info = {}
        print("______________",self.state, reward)
        return self.state, np.float64(reward), done, info



    def reset(self):
        # print(self.observation_space.sample())
        self.state = self.observation_space.sample()
        self.execution_time = exec_steps
        self.current_step = 0
        self.total_power = 0
        return self.state

def exec_main(c_0, c_1):  # main function
    env = Dynamical_Sys(exec_steps, c_0=c_0,c_1=c_1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=15000)
    model.save("./experiment_data/models_" + str("all") + "/dynamics_" + str(c_0) + "___" + str(c_1))


# if __name__ == "__main__":
#     fig, axs = plt.subplots(2)
#     fig.suptitle('power and performance against time')
#     C0_vals = np.linspace(0, 5, 2)
#     C1_vals = np.linspace(0, 5, 2)
#     for i in C0_vals:
#         for l in C1_vals:
#             exec_man(i, l)

env = Dynamical_Sys(exec_time)
check_env(env, warn=True, skip_render_check=True)
# obs = env.reset()
# print(obs)