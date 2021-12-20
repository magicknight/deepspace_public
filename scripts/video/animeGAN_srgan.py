#!/usr/bin/env python3

import numpy as np
import torch
from moviepy.editor import *
from tqdm import tqdm
from pathlib import Path
import shutil
from pprint import pprint
from imageio import imread, imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from srgan import Generator

from imagedataset import get_dataloader

from commontools.setup import config, logger

pprint(config)
# Config
# 原始视频地址
# config.deepspace.original_video = '../data/高中教师.mp4'
# # 提取视频图像的存放地址
# config.deepspace.original_video_img = '../image/高中教师_img/'
# # 合成视频存放地址
# # 添加声音后的视频最终输出地址
# config.deepspace.output_video = './高中教师_audio.mp4'


def save_img(img_array, root, index=None, progress_bar=True, color_mode=None, normalize_mode=None):
    """把 numpy array 按序列保存为图片
    """
    if index is None:
        index = range(img_array.shape[0])
    if progress_bar:
        batches = tqdm(range(img_array.shape[0]), desc="保存图片")
    else:
        batches = range(img_array.shape[0])
    for i in batches:
        if normalize_mode == 'tensorflow':
            result = ((img_array[i] + 1.) * 127.5).clip(0, 255).astype(np.uint8)
        elif normalize_mode == 'torch':
            result = ((img_array[i]) * 255).clip(0, 255).astype(np.uint8)
        imwrite(Path(root) / (str(index[i]) + ".jpg"), result)


# 从视频提取图片
def video_to_image():
    """把视频转换成图片
    """
    """
    # 截取视频中的一段时间
    """
    clip = VideoFileClip(config.deepspace.original_video)

    if config.deepspace.end_time != 0:
        clip = clip.subclip(config.deepspace.start_time,  config.deepspace.end_time)
    if 'resize' in config.deepspace:
        clip = clip.resize(config.deepspace.resize)
    clip.write_images_sequence(config.deepspace.original_video_img + '%03d.jpg', fps=config.deepspace.fps)


def image_to_video():
    """把高分辨率动漫图片转换成视频
    """
    # 读取原始视频
    video_o = VideoFileClip(config.deepspace.original_video)
    # 获取原始视频的音频部分
    audio_o = video_o.audio
    # 获取图片
    images = sorted(list(Path(config.deepspace.high_video_img).glob('*.jpg')), key=lambda x: int(x.stem))
    images = [str(i) for i in images]
    video_clip = ImageSequenceClip(images, fps=config.deepspace.fps)
    # 指向新生成视频的音频部分
    video_clip = video_clip.set_audio(audio_o)
    # 修改音频部分并输出最终视频
    video_clip.write_videofile(config.deepspace.output_video)


@ torch.no_grad()
def real_to_anime():
    """把真实图片转换成动漫图片
    """
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=config.deepspace.pretrained, device=config.deepspace.device,)
    model.to(config.deepspace.device).eval()
    # 转换后的图片存在numpy array中
    # get data loader
    data_loader = get_dataloader(Path(config.deepspace.original_video_img), batch=config.deepspace.anime_batch, normalize_mode='tensorflow')
    # 开始转换
    for images, index in tqdm(data_loader, desc="真实图片转换成动漫图片"):
        # 获取图片
        images = images.to(config.deepspace.device)
        # 转换图片
        try:
            result = model(images, False)  # 转换动漫风格
        except Exception as e:
            print(e)
            continue
        # 保存图片
        save_img(result.permute(0, 2, 3, 1).cpu().numpy(), config.deepspace.anime_video_img, index=index.cpu().numpy(), progress_bar=False, color_mode='RGB', normalize_mode='tensorflow')
    return


@ torch.no_grad()
def anime_to_high_resolution():
    """把低分辨率的动漫图片转换成高分辨率的动漫图片
    """
    # 获取图片总数
    model = Generator()
    model.load_state_dict(torch.load(config.deepspace.model_path), strict=True)
    model.to(config.deepspace.device).eval()
    # 转换后的图片存在numpy array中
    result_array = []
    # get data loader
    data_loader = get_dataloader(Path(config.deepspace.anime_video_img), batch=config.deepspace.hs_batch, normalize_mode='torch')
    # 开始转换
    for images, index in tqdm(data_loader, desc="低分辨率的动漫图片转换成高分辨率"):
        # 获取图片
        images = images.to(config.deepspace.device)
        # 转换图片
        try:
            result = model(images)
        except Exception as e:
            print(e)
            continue
        # 保存图片
        save_img(result.permute(0, 2, 3, 1).cpu().numpy(), config.deepspace.high_video_img, index=index.cpu().numpy(), progress_bar=False, normalize_mode='torch')
        # result_array.append(result.permute(0, 2, 3, 1).cpu().numpy())
    # list to numpy array
    # result_array = np.concatenate(result_array, axis=0)
    # 保存图片
    # save_img(result_array, config.deepspace.high_video_img)
    return


if __name__ == '__main__':

    # 第一步：视频->图像
    Path(config.deepspace.original_video_img).mkdir(parents=True, exist_ok=True)
    video_to_image()

    # 第二步：转换为动漫效果
    Path(config.deepspace.anime_video_img).mkdir(parents=True, exist_ok=True)
    real_to_anime()

    # 第三步：转换为高分辨率图片
    if 'scale' in config.deepspace and config.deepspace.scale > 1:
        Path(config.deepspace.high_video_img).mkdir(parents=True, exist_ok=True)
        anime_to_high_resolution()
    else:
        config.deepspace.high_video_img = config.deepspace.anime_video_img

    # 第四步：合成视频
    image_to_video()

    # 第五步：删除临时文件
    # shutil.rmtree(config.deepspace.original_video_img)
    # shutil.rmtree(config.deepspace.anime_video_img)
    # shutil.rmtree(config.deepspace.high_video_img)