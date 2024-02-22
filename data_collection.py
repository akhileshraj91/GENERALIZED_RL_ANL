#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import collections
import concurrent.futures
import contextlib
import csv
import logging
import logging.config
import pathlib
import time
import typing
import uuid

import cerberus
import ruamel.yaml

import nrm.tooling as nrm


# frequency (in hertz) for RAPL sensor polling
RAPL_SENSOR_FREQ = 1

# maximum number of tries to get extra sensors definitions
CPD_SENSORS_MAXTRY = 5

# maximum time (in second) to wait for a metric read (daemon.upstream_recv)
METRIC_COLLECTION_TIMEOUT = 0.1


# logging configuration  ######################################################

LOGGER_NAME = 'identification-runner'

LOGS_LEVEL = 'INFO'

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
            'filename': f'/tmp/{LOGGER_NAME}.log',
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


# experiment plan validation/load  ############################################

XP_PLAN_VERSION = 1  # current version of experiment plan format

XP_PLAN_ACTION_SUBSCHEMAS = {
    'set_rapl_powercap': {
        'items': (
            {
                'type': 'float',
                'min': 0,
            },
        ),
    },
}

XP_PLAN_SCHEMA = {
    'version': {
        'type': 'integer',
        'min': 1,
        'required': True,
    },
    'actions': {
        'type': 'list',
        'required': True,
        'empty': True,
        'schema': {
            'type': 'dict',
            'allow_unknown': True,
            'oneof_schema': tuple(
                {
                    'time': {
                        'type': 'float',
                        'min': 0,
                        'required': True,
                    },
                    'action': {
                        'allowed': (
                            action,
                        ),
                        'required': True,
                    },
                    'args': {
                        'type': 'list',
                        'required': True,
                        **args_schema
                    },
                }
                for action, args_schema in XP_PLAN_ACTION_SUBSCHEMAS.items()
            )
        },
    },
}


class PowercapAction(typing.NamedTuple):
    time: float
    powercap: float

    @classmethod
    def from_action(cls, action):
        return cls(time=action['time'], powercap=action['args'][0])


XP_PLAN_ACTION_CTORS = {
    'set_rapl_powercap': PowercapAction.from_action,
}
assert set(XP_PLAN_ACTION_SUBSCHEMAS).issubset(XP_PLAN_ACTION_CTORS)


def read_experiment_plan(filepath):
    yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
    raw_config = yaml_parser.load(pathlib.Path(filepath))

    # check experiment plan follows the defined schema
    validator = cerberus.Validator(schema=XP_PLAN_SCHEMA)
    config = validator.validated(raw_config)

    if config is None:
        raise argparse.ArgumentTypeError('bogus experiment plan')

    if config['version'] != XP_PLAN_VERSION:
        raise argparse.ArgumentTypeError(
            f'invalid version of experiment plan format, expected version {XP_PLAN_VERSION}'
        )

    # structure experiment plan data structure
    plan = collections.defaultdict(list)
    for action in config['actions']:
        plan[action['action']].append(
            XP_PLAN_ACTION_CTORS[action['action']](action)
        )

    # check there are no conflicting actions at a given time
    for action, events in plan.items():
        already_defined_times = set()
        for event in events:
            if event.time in already_defined_times:
                raise argparse.ArgumentTypeError(
                    f'conflicting "{action}" actions at time {event.time}'
                )
            already_defined_times.add(event.time)

    return dict(plan)


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
        msg_type: stack.enter_context(open(f'/tmp/dump_{msg_type}.csv', 'w'))
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


# helper functions  ###########################################################

# usage of asyncio:
#
# asyncio is used to interleave interactions with NRM daemon (namely request
# and listen actions).
# Using asyncio does not bring any parallelism: it is a convenient way to
# schedule actions according to the experiment plan.
# The daemon object is not protected by any lock: this might be buggy.

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


async def update_powercap(daemon, rapl_actuators, delay, powercap):
    await asyncio.sleep(delay)
    enforce_powercap(daemon, rapl_actuators, powercap)


async def execute_experiment_plan(plan, daemon, rapl_actuators):
    rapl_actions = (
        update_powercap(daemon, rapl_actuators, delay, powercap)
        for delay, powercap in plan['set_rapl_powercap']
    )
    await asyncio.gather(*rapl_actions)


async def collect_metrics(daemon, csvwriters):
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while not daemon.all_finished():
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(pool, daemon.upstream_recv),
                    timeout=METRIC_COLLECTION_TIMEOUT
                )
                dump_upstream_msg(csvwriters, msg)
            except asyncio.TimeoutError:
                logger.error('metric collection timeout')

    # use `to_thread` with Python 3.9+
    # while not daemon.all_finished():
    #     try:
    #         msg = await asyncio.wait_for(
    #             asyncio.to_thread(daemon.upstream_recv)
    #             timeout=METRIC_COLLECTION_TIMEOUT
    #         )
    #         dump_upstream_msg(csvwriters, msg)
    #     except asyncio.TimeoutError:
    #         pass  # treat timeout


async def do_daemon_ios(plan, daemon, rapl_actuators, csvwriters):
    metric_collection_task = asyncio.create_task(
        # mandatory task as we want to collect metrics as long as the
        # application is running
        collect_metrics(daemon, csvwriters),
        # name='read'
    )
    experiment_plan_task = asyncio.create_task(
        # optional task as it is pointless to change the powercap once the
        # application finished its execution
        execute_experiment_plan(plan, daemon, rapl_actuators),
        # name='write'
    )

    # - mandatory_tasks is the set of all tasks we *do* want to wait for
    # - optional_tasks is the set of tasks that may be cancelled once all tasks
    #   in mandatory_tasks have finished
    mandatory_tasks = {metric_collection_task}
    optional_tasks = {experiment_plan_task}

    # wait until all tasks of mandatory_tasks have finished
    alldone, pending = set(), optional_tasks | mandatory_tasks
    while not alldone.issuperset(mandatory_tasks):
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        alldone |= done

    # cancel remaining tasks of optional_tasks
    for task in pending:
        task.cancel()

    if experiment_plan_task in pending:
        logger.warning('experiment plan partially executed')


async def launch_application(plan, daemon_cfg, workload_cfg, *, libnrm, sleep_duration=0.5):
    with nrm.nrmd(daemon_cfg) as daemon:
        # collect RAPL actuators
        rapl_actuators = collect_rapl_actuators(daemon)

        # collect workload-independent sensors (e.g., RAPL)
        sensors = daemon.req_cpd().sensors()
        logger.info(f'daemon_sensors={sensors}')

        # XXX: trigger actions with time == 0

        # launch workload
        logger.info('launch workload')
        daemon.run(**workload_cfg)

        # retrieve definition of extra sensors if required
        if libnrm:
            app_sensors = update_sensors_list(daemon, sensors, sleep_duration=sleep_duration)
            if not app_sensors:
                logger.critical('failed to get application-specific sensors')
                raise RuntimeError('Unable to get application-specific sensors')
            logger.info(f'app_sensors={app_sensors}')

        # dump sensors values while waiting for the end of the execution
        with contextlib.ExitStack() as stack:
            # each message type is dumped into its own csv file
            # each csv file is created with open
            # we combine all open context managers thanks to an ExitStack
            csvwriters = initialize_csvwriters(stack)

            await do_daemon_ios(plan, daemon, rapl_actuators, csvwriters)


# main script  ################################################################

def cli(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--enable-libnrm',
        action='store_true',
        dest='libnrm',
        help='Enable libnrm instrumentation.',
    )
    parser.add_argument(
        'plan',
        type=read_experiment_plan,
        metavar='plan-filepath',
        help='Path of experiment plan.',
    )

    options, cmd = parser.parse_known_args(args)
    return options, cmd


def run(options, cmd):
    # daemon configuration
    daemon_cfg = {
        'raplCfg': {
            'raplActions': [
                {'microwatts': 1_000_000 * powercap}
                for powercap in set(
                    action.powercap
                    for action in options.plan['set_rapl_powercap']
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
            'app': {},
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

    # configure libnrm instrumentation if required
    if options.libnrm:
        workload_cfg['manifest']['app']['instrumentation'] = {
            'ratelimit': {'hertz': 1_000_000},
        }

    logger.info(f'daemon_cfg={daemon_cfg}')
    logger.info(f'workload_cfg={workload_cfg}')
    asyncio.run(
        launch_application(
            options.plan,
            daemon_cfg,
            workload_cfg,
            libnrm=options.libnrm,
        )
    )
    logger.info('successful execution')


if __name__ == '__main__':
    options, cmd = cli()
    run(options, cmd)
