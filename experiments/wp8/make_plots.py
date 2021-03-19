import numpy as np
from pathlib import Path
from imageio import imread
from matplotlib import pyplot as plt, axis
from tqdm import tqdm

from deepspace.utils.io import showImagesHorizontally
from deepspace.config.config import config, logger


def make_plots():
    root = Path(config.settings.project_root) / config.summary.name / 'out' / 'test'
    diff_path = root / 'diff'
    input_path = root / 'input'
    recon_path = root / 'recon'
    input_paths = list(input_path.glob('**/*.png'))
    recon_paths = list(recon_path.glob('**/*.png'))
    diff_paths = list(diff_path.glob('**/*.png'))
    output_dir = root / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    # set figure size
    plt.rcParams['figure.dpi'] = 200  # 200 e.g. is really fine, but slower

    for index in tqdm(range(len(recon_paths))):
        showImagesHorizontally([input_paths[index], recon_paths[index], diff_paths[index]], path=output_dir / ('inputs' + str(index) + '.png'))
        diff_image = imread(diff_paths[index])
        input_image = imread(input_paths[index])
        defect_y, defect_x = np.where(diff_image == diff_image.max())
        implot = plt.imshow(input_image, cmap='gray', zorder=1)
        plt.scatter(x=defect_x, y=defect_y, c='r', s=10, zorder=2)
        plt.axis('off')
        plt.savefig(output_dir / (str(index) + '.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    make_plots()
