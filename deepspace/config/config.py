import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import toml
from easydict import EasyDict
from pprint import pprint
from pathlib import Path
import argparse


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)


def setup_logging(log_dir):
    """setup logging

    Args:
        log_dir (string): log dir

    """
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)
    return main_logger


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def get_config_from_toml(toml_file):
    """
    Get the config from a toml file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(toml_file, 'r') as config_file:
        try:
            config_dict = toml.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good toml file")
            exit(-1)


def process_config(config_file):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    # get ext, go toml or json
    _, config_file_ext = os.path.splitext(config_file)
    if config_file_ext == '.toml':
        config, _ = get_config_from_toml(config_file)
    elif config_file_ext == '.json':
        config, _ = get_config_from_json(config_file)
    else:
        print("INVALID config file extension.. Please provide a good config file in toml or json")
        exit(-1)
    print(" THE Configuration of your experiment ..")
    pprint(config)

    summary = config.summary
    settings = config.settings
    config.swap = {}
    # making sure that you have provided the name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(summary.name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the experiment name in json file..")
        exit(-1)

    # create some important directories to be used for that experiment.
    if 'project_root' not in settings:
        settings.project_root = Path('.') / "experiments"
    root = Path(settings.project_root)
    config.swap.work_space = root / summary.name
    config.swap.summary_dir = config.swap.work_space / "summaries/"
    config.swap.checkpoint_dir = config.swap.work_space / "checkpoints/"
    config.swap.out_dir = config.swap.work_space / "out/"
    config.swap.log_dir = config.swap.work_space / "logs/"
    create_dirs([config.swap.summary_dir, config.swap.checkpoint_dir, config.swap.out_dir, config.swap.log_dir])

    # setup logging in the project
    logger = setup_logging(config.swap.log_dir)

    logging.getLogger().info("Hi, This is deepspace.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")
    return config, logger


def get_config():
    """
    -Get the args
    -Capture the config file
    -Process the json config passed
    -Merge configs
    """
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config, logger = process_config(args.config)
    return config, logger


config, logger = get_config()


if __name__ == "__main__":
    process_config('configs/abnormal_dataset.toml')
