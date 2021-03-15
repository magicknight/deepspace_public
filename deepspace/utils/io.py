import os
from toml import dump
from pathlib import Path


def save_settings(config, path):
    with open(path, 'w+') as f:
        dump(config, f)


if __name__ == '__main__':
    pass
    # save_settings(Path.home() / 'temp' / 'settings.toml')
