from ruamel.yaml import YAML
import cerberus
import ruamel
import pathlib
import random

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

file_path = './stairs.yaml'
validator = cerberus.Validator(schema=XP_PLAN_SCHEMA)
yaml_parser = ruamel.yaml.YAML(typ='safe', pure=True)
raw_config = yaml_parser.load(pathlib.Path(file_path))

config = validator.validated(raw_config)


yaml = YAML()

# Validate data against schema
data = {
    'version': XP_PLAN_VERSION,
    'actions': [
        {'time': 0, 'action': 'set_rapl_powercap', 'args': [120]},
    ]
}
T = 5
new_time = 0
for i in range(0,5000):
    new_time += T
    act = random.randint(40,120)
    data['actions'].append({'time': new_time, 'action': 'set_rapl_powercap', 'args': [act]})

validator.allow_unknown = True
validator.schema = XP_PLAN_SCHEMA
is_valid = validator.validate(data)
# Dump the data to YAML
with open('./new_stairs.yaml','w') as new_file:
    yaml.dump(data, new_file)


