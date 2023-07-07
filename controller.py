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

import cerberus
import ruamel.yaml

import nrm.tooling as nrm
import os

# experiment_dir = '/home/cc/ANL_comparison/experiment_data/PI_control/'
# if not os.path.isdir(experiment_dir):
#     os.mkdir(experiment_dir)

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
                'filename': f'./experiment_data/PI_control/{WORKLOAD}/{LOGGER_NAME}.log',
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
        msg_type: stack.enter_context(open(f'./experiment_data/PI_control/{WORKLOAD}/dump_{msg_type}.csv', 'w'))
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


# controller logic & configuration  ###########################################

# The system is approximated as a first order linear system.
# The controller is implemented as a Proportional-Integral (PI) controller.
#
# The configuration of the controller is divided in three sections:
#
# 1. RAPL characterization  -----
#
#   The effective power consumption does not exactly match the powercap
#   command.
#   The deviation is modeled as an affine relation:
#       power consumption = f(powercap)
#                         = config['rapl']['slope'] * powercap + config['rapl']['offset']
#
#   where:
#     - config['rapl']['slope'] is unitless
#     - config['rapl']['offset'] is in Watt
#
#
# 2. model parameters  -----
#
#   The system is modeled with the following equation:
#       progress = f(powercap)
#                = config['model']['gain'] * (1 - exp(-config['model']['alpha'] * (powercap - config['model']['beta'])))
#
#   where:
#     - config['model']['alpha'] is in 1/Watt
#     - config['model']['beta'] is in Watt
#     - config['model']['gain'] is in Hertz
#
#   The benchmark/cluster modelization impacts config['model']['alpha'] and
#   config['model']['beta'] parameters.
#
#   The relationship between the powercap and the progress (i.e., progress as a
#   function of powercap) is modeled with a first-order linear approximation.
#   It is configured by the config['model']['gain'] and
#   config['model']['time-constant'] parameters.
#
#
# 3. controller parameters  -----
#
#   The controller is configured with the following parameters:
#    - config['controller']['setpoint'] (unitless):
#        A value in the continuous interval [0, 1] representing the level of
#        performance to achieve (as a proportion of the maximum performance).
#    - config['controller']['response-time'] (in second):
#        5% response time (defined as 3·τ)
#    - config['controller']['power-range'] (in Watt):
#        A pair (low, high) — where low < high — of value representing the
#        range of powercap values the controller is allowed to use.


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

    def control(self, csvwriters):
        # print(self.daemon,self.daemon.all_finished())
        while not self.daemon.all_finished():
            msg = self.daemon.upstream_recv()  # blocking call
            # logger.info(f"The recieved messages on the control side are {msg}")
            dump_upstream_msg(csvwriters, msg)
            (msg_type, payload), = msg.items()  # single-key dict destructuring
            # dispatch to relevant logic
            print(f"____________,{msg}")
            if msg_type == 'pubProgress':
                self._update_progress(payload)
            elif msg_type == 'pubMeasurements':
                self._update_measure(payload)

    def _update_progress(self, payload):
        timestamp, _, value = payload
        timestamp *= 1e-6  # convert µs in s
        print(self.heartbeat_timestamps)
        self.heartbeat_timestamps.append((timestamp,value))
        #self.heartbeat_timestamps.append(timestamp)

    @staticmethod
    def _estimate_progress(heartbeat_timestamps):
        """Estimate the heartbeats' frequency given a list of heartbeats' timestamps."""
        #return statistics.median(
            #1 / (second - first)
            #for first, second in zip(heartbeat_timestamps, heartbeat_timestamps[1:])
        #)
        #print(heartbeat_timestamps)
        # logger.info(f"The heartbeat_timestamps are: {heartbeat_timestamps}")
        return statistics.median(((second[1])/ (second[0] - first[0]))
            for first, second in zip(heartbeat_timestamps, heartbeat_timestamps[1:]))

    def _update_measure(self, payload):
        timestamp, measures = payload
        timestamp *= 1e-6  # convert µs in s
        # logger.info(f"measures : {measures}")
        # logger.info(f"each measure is {[da['sensorID'] for da in measures]}")
        for data in measures:
            # print(data,data['sensorID'])
            if data['sensorID'].startswith('RaplKey'):
                # print("_____________________________________")
                # print(self.heartbeat_timestamps)
                # collect sensors
                window_duration = timestamp - self.rapl_window_timestamp
                progress_estimation = self._estimate_progress(self.heartbeat_timestamps)
                # logger.info(f"The heartbeat_timestamps in update measure are: {self.heartbeat_timestamps}")


                # estimate current error
                error = self.progress_setpoint - progress_estimation

                # compute command with linear equation
                self.powercap_linear = \
                        window_duration * self._integral_gain * error + \
                        self._proportional_gain * (error - self.prev_error) + \
                        self.powercap_linear

                # thresholding (ensure powercap remains in controller power range)
                #   this can be done in the linear space as the variable change
                #   is monotic (increasing as self._model_alpha > 0)
                self.powercap_linear = max(
                    min(
                        self.powercap_linear,
                        self._powercap_linear_max
                    ),
                    self._powercap_linear_min
                )

                # delinearize to get actual actuator value
                powercap = self._delinearize(self.powercap_linear)

                # propagate state
                self.prev_error = error

                # reset monitoring variables for new upcoming RAPL window
                self.rapl_window_timestamp = timestamp
                self.heartbeat_timestamps = self.heartbeat_timestamps[-1:]

                # send command to actuator
                powercap = round(powercap)  # XXX: discretization induced by raplActions
                enforce_powercap(self.daemon, self.rapl_actuators, powercap)

                # we treat all RAPL packages at once, ignore other RAPL sensors
                break


# helper functions  ###########################################################

def update_sensors_list(daemon, known_sensors, *, maxtry=CPD_SENSORS_MAXTRY, sleep_duration=0.5):
    """Update in place the list known_sensors, returns the new sensors."""
    assert isinstance(known_sensors, list)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{known_sensors}")

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

    print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{new_sensors}")

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


def launch_application(config, daemon_cfg, workload_cfg, *, sleep_duration=0.5):
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
            controller.control(csvwriters)


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
        'args': cmd[1:],
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
    launch_application(options.config, daemon_cfg, workload_cfg)
    logger.info('successful execution')


if __name__ == '__main__':
    options, cmd = cli()
    WORKLOAD = cmd[0]
    LOGS_CONF, logger = logs_conf_func()
    run(options, cmd)
