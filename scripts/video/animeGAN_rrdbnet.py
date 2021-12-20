#!/usr/bin/env python3

import cv2
from PIL import Image
from cv2 import normalize
import numpy as np
import os
import torch
from moviepy.editor import *
from torch import save
from tqdm import tqdm
from pathlib import Path
import shutil
from pprint import pprint
from imageio import imread, imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet

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
        if color_mode == 'RGB':
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(root) / (str(index[i]) + ".jpg")), result)
        else:
            imwrite(Path(root) / (str(index[i]) + ".jpg"), result)


# 从视频提取图片
def video_to_image():
    """把视频转换成图片
    """
    cap = cv2.VideoCapture(config.deepspace.original_video)
    config.deepspace.fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
    i = 1
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        else:
            if 'resolution' in config.deepspace:
                frame = cv2.resize(frame, tuple(config.deepspace.resolution))
            if 'resize' in config.deepspace:
                frame = cv2.resize(frame, (0, 0), fx=config.deepspace.resize, fy=config.deepspace.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(config.deepspace.original_video_img + str(i) + ".jpg", frame)
            i += 1
    return


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
    video_clip = ImageSequenceClip(images, fps=config.deepspace.fps, load_images=False)
    # 指向新生成视频的音频部分
    video_clip = video_clip.set_audio(audio_o)
    # 修改音频部分并输出最终视频
    video_clip.write_videofile(config.deepspace.output_video)


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
        with torch.no_grad():
            try:
                result = model(images, False)  # 转换动漫风格
            except Exception as e:
                print(e)
                continue
        # 保存图片
        save_img(result.permute(0, 2, 3, 1).cpu().numpy(), config.deepspace.anime_video_img, index=index.cpu().numpy(), progress_bar=False, color_mode='RGB', normalize_mode='tensorflow')
    return


def anime_to_high_resolution():
    """把低分辨率的动漫图片转换成高分辨率的动漫图片
    """
    # 获取图片总数
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=config.deepspace.scale,)
    model.load_state_dict(torch.load(config.deepspace.model_path)['params_ema'], strict=True)
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
        with torch.no_grad():
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


def cut_video():
    """
    # 截取视频中的一段时间
    """
    video_path = Path(config.deepspace.original_video)
    video_clip = VideoFileClip(config.deepspace.original_video)
    video_clip.subclip(config.deepspace.start_time,  config.deepspace.end_time).write_videofile(str(video_path.parent / (video_path.stem + '_cut.mp4')))
    # config.deepspace.original_video = str(video_path.parent / (video_path.stem + '_cut.mp4'))


if __name__ == '__main__':

    # 第零步：截取视频中的一段时间
    if config.deepspace.end_time != 0:
        # cut_video()
        config.deepspace.original_video = str(Path(config.deepspace.original_video).parent / (Path(config.deepspace.original_video).stem + '_cut.mp4'))
    # 第一步：视频->图像
    Path(config.deepspace.original_video_img).mkdir(parents=True, exist_ok=True)
    # video_to_image()

    # 第二步：转换为动漫效果
    Path(config.deepspace.anime_video_img).mkdir(parents=True, exist_ok=True)
    # real_to_anime()

    # 第三步：转换为高分辨率图片
    if 'scale' in config.deepspace and config.deepspace.scale > 1:
        Path(config.deepspace.high_video_img).mkdir(parents=True, exist_ok=True)
        # anime_to_high_resolution()
    else:
        config.deepspace.high_video_img = config.deepspace.anime_video_img

    # 第四步：合成视频
    image_to_video()

    # 第五步：删除临时文件
    # shutil.rmtree(config.deepspace.original_video_img)
    # shutil.rmtree(config.deepspace.anime_video_img)
    # shutil.rmtree(config.deepspace.high_video_img)