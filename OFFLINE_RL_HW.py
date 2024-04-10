#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import csv
import logging
import logging.config
import math
import pathlib
import statistics
import time
import uuid
import numpy as np
import cerberus
import ruamel.yaml

import nrm.tooling as nrm
from stable_baselines3 import PPO
import train_multimodels as ncna
import os

# WLs = {'ones-stream-full':0,'ones-stream-add':1,'ones-stream-triad':2,'ones-stream-scale':3,'ones-stream-copy':4,'ones-npb-ep':5}

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

# frequency (in hertz) for RAPL sensor polling
RAPL_SENSOR_FREQ = 1

# maximum number of tries to get extra sensors definitions
CPD_SENSORS_MAXTRY = 25


# logging configuration  ######################################################

LOGGER_NAME = 'controller-runner'

LOGS_LEVEL = 'INFO'

def logs_conf_func():
    LOGS_CONF = {
        'version': 1,
        'formatters': {
            'precise': {
                # timestamp is epoch in seconds
                'format': '{created}\u0000{levelname}\u0000{process}\u0000{funcName}\u0000{message}',
                'style': '{',
            },
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': f'./experiment_data/RL_controller/{WORKLOAD}/{LOGGER_NAME}.log',
                'mode': 'w',
                'level': LOGS_LEVEL,
                'formatter': 'precise',
                'filters': [],
            },
        },
        'loggers': {
            LOGGER_NAME: {
                'level': LOGS_LEVEL,
                'handlers': [
                    'file',
                ],
            },
        },
    }

    logging.config.dictConfig(LOGS_CONF)


    logger = logging.getLogger(LOGGER_NAME)
    return LOGS_CONF, logger



# controller configuration validation/load  ###################################

CTRL_CONFIG_VERSION = 1  # current version of controller configuration format

CTRL_CONFIG_SCHEMA = {
    'version': {
        'type': 'integer',
        'min': 1,
        'required': True,
    },
    'rapl': {
        'type': 'dict',
        'required': True,
        'schema': {
            'slope': {
                'type': 'float',
                'required': True,
            },
            'offset': {
                'type': 'float',
                'required': True,
            },
        },
    },
    'model': {
        'type': 'dict',
        'required': True,
        'schema': {
            'alpha': {
                'type': 'float',
                'min': 0,  # thresholding in linear space requires alpha > 0
                'required': True,
            },
            'beta': {
                'type': 'float',
                'required': True,
            },
            'gain': {
                'type': 'float',
                'required': True,
            },
            'time-constant': {
                'type': 'float',
                'min': 0,
                'required': True,
            },
        },
    },
    'controller': {
        'type': 'dict',
        'required': True,
        'schema': {
            'response-time': {
                'type': 'float',
                'min': 0,
                'required': True,
            },
            'setpoint': {
                'type': 'float',
                'min': 0,
                'max': 1,
                'required': True,
            },
            'power-range': {
                'type': 'list',
                'items': [
                    {  # minimum powercap
                        'type': 'float',
                        'min': 0,
                    },
                    {  # maximum powercap
                        'type': 'float',
                        'min': 0,
                    },
                ],
                'required': True,
            },
        },
    },
}


def read_controller_configuration(filepath):
    yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
    raw_config = yaml_parser.load(pathlib.Path(filepath))

    # check controller configuration follows the defined schema
    validator = cerberus.Validator(schema=CTRL_CONFIG_SCHEMA)
    config = validator.validated(raw_config)

    if config is None:
        raise argparse.ArgumentTypeError('bogus controller configuration')

    if config['version'] != CTRL_CONFIG_VERSION:
        raise argparse.ArgumentTypeError(
            f'invalid version of controller configuration format, expected version {CTRL_CONFIG_VERSION}'
        )

    if config['controller']['power-range'][0] >= config['controller']['power-range'][1]:
        raise argparse.ArgumentTypeError('invalid power range')

    return dict(config)


# CSV export  #################################################################

DUMPED_MSG_TYPES = {
    'pubMeasurements',
    'pubProgress',
}

CSV_FIELDS = {
    'common': (
        'msg.timestamp',
        'msg.id',
        'msg.type',
    ),
    'pubMeasurements': (
        'sensor.timestamp',  # time
        'sensor.id',         # sensorID
        'sensor.value',      # sensorValue
    ),
    'pubProgress': (
        'sensor.cmd',    # cmdID
        'sensor.task',   # taskID
        'sensor.rank',   # rankID
        'sensor.pid',    # processID
        'sensor.tid',    # threadID
        'sensor.value',
    ),
}
assert DUMPED_MSG_TYPES.issubset(CSV_FIELDS)


def initialize_csvwriters(stack: contextlib.ExitStack):
    csvfiles = {
        msg_type: stack.enter_context(open(f'./experiment_data/RL_controller/{WORKLOAD}/dump_{msg_type}.csv', 'w'))
        for msg_type in DUMPED_MSG_TYPES
    }

    csvwriters = {
        msg_type: csv.DictWriter(csvfile, fieldnames=CSV_FIELDS['common']+CSV_FIELDS[msg_type])
        for msg_type, csvfile in csvfiles.items()
    }
    for csvwriter in csvwriters.values():
        csvwriter.writeheader()

    return csvwriters


def pubMeasurements_extractor(msg_id, payload):
    timestamp, measures = payload
    for data in measures:
        yield {
            'msg.timestamp': timestamp * 1e-6,  # convert µs in s
            'msg.id': msg_id,
            'msg.type': 'pubMeasurements',
            #
            'sensor.timestamp': data['time'] * 1e-6,  # convert µs in s
            'sensor.id': data['sensorID'],
            'sensor.value': data['sensorValue'],
        }


def pubProgress_extractor(msg_id, payload):
    timestamp, identification, value = payload
    yield {
        'msg.timestamp': timestamp * 1e-6,  # convert µs in s
        'msg.id': msg_id,
        'msg.type': 'pubProgress',
        #
        'sensor.cmd': identification['cmdID'],
        'sensor.task': identification['taskID'],
        'sensor.rank': identification['rankID'],
        'sensor.pid': identification['processID'],
        'sensor.tid': identification['threadID'],
        'sensor.value': value,
    }


def noop_extractor(*_):
    yield from ()


DUMPED_MSG_EXTRACTORS = {
    'pubMeasurements': pubMeasurements_extractor,
    'pubProgress': pubProgress_extractor,
}
assert DUMPED_MSG_TYPES.issubset(DUMPED_MSG_EXTRACTORS)


def dump_upstream_msg(csvwriters, msg):
    msg_id = uuid.uuid4()
    (msg_type, payload), = msg.items()  # single-key dict destructuring
    msg2rows = DUMPED_MSG_EXTRACTORS.get(msg_type, noop_extractor)
    csvwriter = csvwriters.get(msg_type)
    for row in msg2rows(msg_id, payload):
        csvwriter.writerow(row)





class PIController:
    def __init__(self, config, daemon, rapl_actuators):
        self.daemon = daemon
        self.rapl_actuators = rapl_actuators

        # rapl characterization
        self._rapl_slope = config['rapl']['slope']
        self._rapl_offset = config['rapl']['offset']

        # model parameters
        self._model_alpha = config['model']['alpha']
        self._model_beta = config['model']['beta']
        self._model_gain_linear = config['model']['gain']
        self._model_time_constant = config['model']['time-constant']

        # controller parameters
        self._setpoint = config['controller']['setpoint']
        self._response_time = config['controller']['response-time']
        self._proportional_gain = \
                1 / (self._model_gain_linear * self._response_time / 3)
        self._integral_gain = \
                self._model_time_constant * self._proportional_gain
        self._powercap_linear_min = \
                self._linearize(config['controller']['power-range'][0])
        self._powercap_linear_max = \
                self._linearize(config['controller']['power-range'][1])

        # objective configuration (requested system behavior)
        self.progress_setpoint = self._setpoint * self._model_gain_linear * (1 + self._powercap_linear_max)

        # controller initial state
        self.powercap_linear = self._powercap_linear_max
        self.prev_error = 0

        # RAPL window monitoring
        self.rapl_window_timestamp = time.time()  # start of current RAPL window
        self.heartbeat_timestamps = []

    def _linearize(self, value):
        # see model equation
        return -math.exp(-self._model_alpha * (self._rapl_slope * value + self._rapl_offset - self._model_beta))

    def _delinearize(self, value):
        # see model equation
        return (-math.log(-value) / self._model_alpha + self._model_beta - self._rapl_offset) / self._rapl_slope

    def control(self, csvwriters, model_file):
        Q_MATRIX = np.loadtxt(model_file, delimiter=',')
        # self.dynamics = model_file
        self.model =Q_MATRIX
        flag = 0
        # print("_______________________")
        while not self.daemon.all_finished():
            msg = self.daemon.upstream_recv()  # blocking call
            dump_upstream_msg(csvwriters, msg)
            (msg_type, payload), = msg.items()  # single-key dict destructuring
            # dispatch to relevant logic
            #print(len(payload))
            if msg_type == 'pubProgress':
                self._update_progress(payload)
            elif msg_type == 'pubMeasurements':
                self._update_measure(payload)
            else:
                print("continuing")
                continue

    def _update_progress(self, payload):
        timestamp, _, value = payload
        timestamp *= 1e-6  # convert µs in s
        self.heartbeat_timestamps.append((timestamp,value))

    @staticmethod
    def _estimate_progress(heartbeat_timestamps):
        """Estimate the heartbeats' frequency given a list of heartbeats' timestamps."""
        #print(">>>>>>>>>",heartbeat_timestamps)
        if len(heartbeat_timestamps) > 1:
            return statistics.median(((second[1])/ (second[0] - first[0])) for first, second in zip(heartbeat_timestamps, heartbeat_timestamps[1:]))
        else:
            return 0

    def _update_measure(self, payload):
        timestamp, measures = payload
        timestamp *= 1e-6  # convert µs in s
        for data in measures:
            if data['sensorID'].startswith('RaplKey'):
                # window_duration = timestamp - self.rapl_window_timestamp
                progress_estimation = self._estimate_progress(self.heartbeat_timestamps)
                # print(progress_estimation,INDEX)
                obs = progress_estimation
                print(obs)
                action_space = self.model[round(obs),:]
                action = np.argmax(action_space)
                # action, _states = self.model.predict(obs, deterministic=True)
                ab_action = action+40
                powercap = ab_action

                print(powercap)
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>",powercap)
                # self.powercap_linear = max(
                #     min(
                #         self.powercap_linear,
                #         self._powercap_linear_max
                #     ),
                #     self._powercap_linear_min
                # )

                self.rapl_window_timestamp = timestamp
                self.heartbeat_timestamps = self.heartbeat_timestamps[-1:]
                powercap = round(powercap)
                enforce_powercap(self.daemon, self.rapl_actuators, powercap)

                break


# helper functions  ###########################################################

def update_sensors_list(daemon, known_sensors, *, maxtry=CPD_SENSORS_MAXTRY, sleep_duration=0.5):
    """Update in place the list known_sensors, returns the new sensors."""
    assert isinstance(known_sensors, list)

    new_sensors = []
    for _ in range(maxtry):
        new_sensors = [
            sensor
            for sensor in daemon.req_cpd().sensors()
            if sensor not in known_sensors
        ]
        if new_sensors:
            break  # new sensors have been retrieved
        time.sleep(sleep_duration)

    known_sensors.extend(new_sensors)  # extend known_sensors in place
    return new_sensors


def enforce_powercap(daemon, rapl_actuators, powercap):
    # for each RAPL actuator, create an action that sets the powercap to powercap
    set_pcap_actions = [
        nrm.Action(actuator.actuatorID, powercap)
        for actuator in rapl_actuators
    ]

    logger.info(f'set_pcap={powercap}')
    daemon.actuate(set_pcap_actions)


def collect_rapl_actuators(daemon):
    # the configuration of RAPL actuators solely depends on the daemon: there
    # is no need to wait for the application to start
    cpd = daemon.req_cpd()

    # get all RAPL actuator (XXX: should filter on tag rather than name)
    rapl_actuators = list(
        filter(
            lambda a: a.actuatorID.startswith('RaplKey'),
            cpd.actuators()
        )
    )
    logger.info(f'rapl_actuators={rapl_actuators}')
    return rapl_actuators


def launch_application(config, daemon_cfg, workload_cfg, RLM, *, sleep_duration=0.5):
    #print(">>>>>>>>>>>>>>>>>>>>",RLM)
    with nrm.nrmd(daemon_cfg) as daemon:
        # collect RAPL actuators
        rapl_actuators = collect_rapl_actuators(daemon)

        # collect workload-independent sensors (e.g., RAPL)
        sensors = daemon.req_cpd().sensors()
        logger.info(f'daemon_sensors={sensors}')

        # launch workload
        logger.info('launch workload')
        daemon.run(**workload_cfg)

        # retrieve definition of extra sensors (libnrm progress, …)
        app_sensors = update_sensors_list(daemon, sensors, sleep_duration=sleep_duration)
        if not app_sensors:
            logger.critical('failed to get application-specific sensors')
            raise RuntimeError('Unable to get application-specific sensors')
        logger.info(f'app_sensors={app_sensors}')

        with contextlib.ExitStack() as stack:
            # each message type is dumped into its own csv file
            # each csv file is created with open
            # we combine all open context managers thanks to an ExitStack
            csvwriters = initialize_csvwriters(stack)

            controller = PIController(config, daemon, rapl_actuators)
            controller.control(csvwriters, RLM)


# main script  ################################################################

def cli(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=read_controller_configuration,
        metavar='configuration-filepath',
        help='Path of controller configuration.',
    )

    options, cmd = parser.parse_known_args(args)

    return options, cmd


def run(options, cmd):
    RLM = cmd[-1]
    # daemon configuration
    daemon_cfg = {
        'raplCfg': {
            'raplActions': [  # XXX: check with Valentin for continuous range
                {'microwatts': 1_000_000 * powercap}
                for powercap in range(
                    round(options.config['controller']['power-range'][0]),
                    round(options.config['controller']['power-range'][1]) + 1
                )
            ],
        },
        'passiveSensorFrequency': {
            # this configures the frequency of all 'passive' sensors, we only
            # use RAPL sensors here
            'hertz': RAPL_SENSOR_FREQ,
        },
        # 'verbose': 'Debug',
        # 'verbose': 'Info',
        'verbose': 'Error',
    }

    # workload configuration (i.e., app description + manifest)
    workload_cfg = {
        'cmd': cmd[0],
        'args': cmd[1:-1],
        'sliceID': 'sliceID',  # XXX: bug in pynrm/hnrm if missing or None (should be generated?)
        'manifest': {
            'app': {
                # configure libnrm instrumentation
                'instrumentation': {
                    'ratelimit': {'hertz': 100_000_000},
                },
            },
        },
    }
    # NB: if we want to keep the output of the launched command when run in
    # detached mode (which is the case with the Python API), we can wrap the
    # call in a shell
    #     {
    #         'cmd': 'sh',
    #         'args': ['-c', f'{command} >stdout 2>stderr'],
    #     }
    #
    # Uncomment the lines below to wrap the call:
    # workload_cfg.update(
    #     cmd='sh',
    #     args=['-c', ' '.join(cmd), '>/tmp/stdout', '2>/tmp/stderr'],
    # )

    logger.info(f'daemon_cfg={daemon_cfg}')
    logger.info(f'workload_cfg={workload_cfg}')
    launch_application(options.config, daemon_cfg, workload_cfg, RLM)
    logger.info('successful execution')


if __name__ == '__main__':
    options, cmd = cli()
    WORKLOAD = cmd[0]
    INDEX = APPLICATIONS.index(WORKLOAD)
    #print(INDEX,WORKLOAD)
    LOGS_CONF, logger = logs_conf_func()
    run(options, cmd)
