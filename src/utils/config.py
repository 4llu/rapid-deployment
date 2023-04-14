import os
import ruamel.yaml as yaml


def setup_config(config_name):
    # Determine config path
    abs_path = os.path.dirname(__file__)

    path = os.path.join(
        abs_path, os.pardir, os.pardir, "configs", *(config_name + ".yaml").split("/")
    )

    config = None
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config
