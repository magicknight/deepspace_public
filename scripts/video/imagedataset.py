# pytorch image dataloader
#     """
#     # 创建一个视频读取对象
#     vid = imageio.get_reader(config.deepspace.img2video, 'ffmpeg')
#     # 获取视频的帧数
#     length = vid.get_length()
#     # 创建一个图片写入对象
#     writer = imageio.get_writer(config.deepspace.output_video, fps=vid.get_meta_data()['fps'])
#     # 循环读取视频帧
#     for i in range(length):
#         # 读取视频帧
#         frame = vid.get_data(i)
#         # 写入图片
#         writer.append_data(frame)
#         # 显示视频帧
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     # 释放资源
#     vid.close()
#     writer.close()
#     cv2.destroyAllWindows()
#     return
from imageio import imread
from matplotlib import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Compose

from commontools.setup import config, logger


class ImageDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 获取所有图片的路径， 并且按照数字顺序排序
        self.imgs = sorted(list(self.root.glob('*.jpg')), key=lambda x: int(x.stem))
        # self.imgs.sort()
        self.length = len(self.imgs)
        self.logger = logger
        self.logger.info('ImageDataset: {}'.format(self.length))

    def __getitem__(self, index):
        # img = read_image(str(self.imgs[index])).to(torch.float32)
        img = imread(str(self.imgs[index]))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, index

    def __len__(self):
        return self.length


def get_dataloader(root, batch, normalize_mode=None):
    if normalize_mode is 'torch':
        # normalize = Normalize(mean=[0, 0, 0], std=[255, 255, 255])
        normalize = Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    elif normalize_mode is 'tensorflow':
        # normalize = Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transforms = Compose([ToTensor(), normalize, ])
    dataset = ImageDataset(root, transforms=transforms)
    # if train on tpu
    if config.common.distribute:
        from torch_xla.core import xla_model
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=xla_model.xrt_world_size(),
            rank=xla_model.get_ordinal(),
            shuffle=config.deepspace.shuffle,
        )
    else:
        train_sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=config.deepspace.shuffle,
        sampler=train_sampler,
        drop_last=config.deepspace.drop_last,
        num_workers=config.deepspace.num_workers,
    )
    return dataloader
