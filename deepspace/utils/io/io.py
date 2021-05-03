from imageio import imread
from matplotlib import pyplot as plt
import tqdm
import logging
import os
from toml import dump
from pathlib import Path
import torch
import shutil


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


def showImagesHorizontally(list_of_files, path=None):
    """show multiple images horizontally

    Args:
        list_of_files (list): a list of image paths

    Returns:
        figure: a figure referencer.
    """
    fig = plt.figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a = fig.add_subplot(1, number_of_files, i + 1)
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
        image = imread(list_of_files[i])
        plt.imshow(image, cmap='Greys_r')
        plt.axis('off')
        if path is not None:
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def show_images(list_of_images, path=None):
    """show multiple images horizontally

    Args:
        list_of_images (list): a list of images

    Returns:
        figure: a figure referencer.
    """
    fig = plt.figure()
    number_of_files = len(list_of_images)
    for i in range(number_of_files):
        a = fig.add_subplot(1, number_of_files, i + 1)
        plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0., hspace=0.)
        # image = imread(list_of_images[i])
        plt.imshow(list_of_images[i], cmap='Greys_r')
        plt.axis('off')
        if path is not None:
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def save_checkpoint(agent, filename, is_best=False) -> None:
    """io function for saveing checkpoint during training

    Args:
        state (dict): a dict contains key and values wanted to be svaed
        filename (pathlib Path): a path to the checkpoint file
        is_best (bool, optional): is it the best performance model. Defaults to False.
    """
    state = {}
    for key in agent.checkpoint:
        if '.' in key:
            name, attr = key.split('.')
            state[key] = getattr(getattr(agent, name), attr)()
        else:
            state[key] = getattr(agent, key)
            # state['state_dict'] = {k: v.state_dict() for k, v in state['state_dict'].items()}
            # Save the state
    torch.save(state, filename)
    # If it is the best copy it to another file 'model_best.pth.tar'
    if is_best:
        shutil.copyfile(filename, filename.parent / 'model_best.pth.tar')


def load_checkpoint(agent, filename) -> None:
    """an io function to load checkpoint

    Args:
        agent (deepspace agent): a deepspace agent object
        filename (pathlib Path): a path to the checkpoint file
    """
    checkpoint = torch.load(filename)
    for key, value in checkpoint.items():
        if '.' in key:
            name, attr = key.split('.')
            agent[name].load_state_dict(value)
        else:
            agent[key] = value
    # for k, v in checkpoint['number'].items():
    #     agent[k] = v
    #     # property = getattr(agent, k)
    #     # property = v
    # for k, v in checkpoint['state_dict'].items():
    #     # getattr(agent, k).load_state_dict(v)
    #     agent[k].load_state_dict(v)


if __name__ == '__main__':
    pass
    # save_settings(Path.home() / 'temp' / 'settings.toml')
