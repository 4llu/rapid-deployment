import os
from ruamel.yaml import YAML


def setup_config(config_name, config_override_name=None):
    # Determine config path
    abs_path = os.path.dirname(__file__)
    path = os.path.join(abs_path, os.pardir, os.pardir, "configs", *(config_name + ".yaml").split("/"))

    yaml = YAML()

    config = None
    with open(path) as stream:
        config = yaml.load(stream)

    if not config_override_name is None:
        override_path = os.path.join(
            abs_path, os.pardir, os.pardir, "configs", *(config_override_name + ".yaml").split("/")
        )

        config_override = None
        with open(override_path) as stream:
            config_override = yaml.load(stream)

        config.update(config_override)

    return config
