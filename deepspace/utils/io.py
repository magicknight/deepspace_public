from imageio import imread
from matplotlib.pyplot import figure, imshow, axis
import tqdm
import logging
import os
from toml import dump
from pathlib import Path


def save_settings(config, path):
    """save current configs

    Args:
        config (Dict): a dict of config
        path (string or Path): path of save file
    """
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


def showImagesHorizontally(list_of_files):
    """show multiple images horizontally

    Args:
        list_of_files (list): a list of image paths

    Returns:
        figure: a figure referencer.
    """
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a = fig.add_subplot(1, number_of_files, i+1)
        image = imread(list_of_files[i])
        imshow(image, cmap='Greys_r')
        axis('off')
    return fig


if __name__ == '__main__':
    pass
    # save_settings(Path.home() / 'temp' / 'settings.toml')
