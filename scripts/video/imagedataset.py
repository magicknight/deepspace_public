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
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import ToPILImage

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
        img = read_image(str(self.imgs[index])).to(torch.float32)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, index

    def __len__(self):
        return self.length


def get_dataloader(root, batch, normalize_mode=None):
    if normalize_mode is 'torch':
        normalize = Normalize(mean=[0, 0, 0], std=[255, 255, 255])
    elif normalize_mode is 'tensorflow':
        normalize = Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
    dataset = ImageDataset(root, transforms=normalize)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=config.deepspace.shuffle, num_workers=config.deepspace.num_workers)
    return dataloader
