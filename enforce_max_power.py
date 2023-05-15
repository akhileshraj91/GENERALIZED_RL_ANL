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
import os
import cerberus
import ruamel.yaml
import nrm.tooling as nrm

RAPL_SENSOR_FREQ = 1

CPD_SENSORS_MAXTRY = 5


CTRL_CONFIG_VERSION = 1 
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
                'min': 0,  
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


def enforce_powercap(daemon, rapl_actuators, powercap):
    print(f"enforcing the power {powercap}")
    set_pcap_actions = [
        nrm.Action(actuator.actuatorID, powercap)
        for actuator in rapl_actuators
    ]
    daemon.actuate(set_pcap_actions)


def collect_rapl_actuators(daemon):
    cpd = daemon.req_cpd()

    rapl_actuators = list(
        filter(
            lambda a: a.actuatorID.startswith('RaplKey'),
            cpd.actuators()
        )
    )
    return rapl_actuators



def launch_application(config, daemon_cfg):
    with nrm.nrmd(daemon_cfg) as daemon:
        rapl_actuators = collect_rapl_actuators(daemon)
        sensors = daemon.req_cpd().sensors()
        enforce_powercap(daemon,rapl_actuators,200)

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
    daemon_cfg = {
        'raplCfg': {
            'raplActions': [ 
                {'microwatts': 1_000_000 * powercap}
                for powercap in range(
                    round(options.config['controller']['power-range'][0]),
                    round(options.config['controller']['power-range'][1]) + 1
                )
            ],
        },
        'passiveSensorFrequency': {
            'hertz': RAPL_SENSOR_FREQ,
        },
        'verbose': 'Debug',
        'verbose': 'Error',
    }
    launch_application(options.config, daemon_cfg)




if __name__ == '__main__':
    options, cmd = cli()
    run(options, cmd)

