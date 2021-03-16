import tqdm
import logging
import os
from toml import dump
from pathlib import Path


def save_settings(config, path):
    with open(path, 'w+') as f:
        dump(config, f)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


if __name__ == '__main__':
    pass
    # save_settings(Path.home() / 'temp' / 'settings.toml')
