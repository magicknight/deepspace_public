#!/usr/bin/env python3

import cv2
from PIL import Image
import numpy as np
import os
import torch
from moviepy.editor import *
from tqdm import tqdm
from pathlib import Path
import shutil
from pprint import pprint

from realesrgan import RealESRGANer
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
# config.deepspace.img2video = '../output/高中教师.mp4'
# # 添加声音后的视频最终输出地址
# config.deepspace.output_video = './高中教师_audio.mp4'

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
            cv2.imwrite(config.deepspace.original_video_img + str(i) + ".jpg", frame)
            i += 1
    return


def image_to_video():
    """把高分辨率动漫图片转换成视频
    """
    # 获取图片总数
    file_list = os.listdir(config.deepspace.high_video_img)
    img_num = len(file_list)

    # 任选一张图片查看高度和宽度
    # result = model.style_transfer(images=[cv2.imread(os.path.join(config.deepspace.original_video_img, file_list[0]))])
    image = cv2.imread(os.path.join(config.deepspace.high_video_img, file_list[0]))
    height = image.shape[0]
    width = image.shape[1]

    # 创建一个VideoWriter对象，并视频文件名，视频帧率，视频尺寸
    video = cv2.VideoWriter(config.deepspace.img2video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), config.deepspace.fps, (width, height))

    for i in tqdm(range(img_num)):
        try:
            f_name = str(i + 1) + '.jpg'
            item = os.path.join(config.deepspace.high_video_img, f_name)
            image = cv2.imread(item)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)  # 把图片写进视频
        except Exception as e:
            print(e)
            continue
    video.release()


def real_to_anime():
    """把真实图片转换成动漫图片
    """
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=config.deepspace.pretrained, device=config.deepspace.device,)
    model.to(config.deepspace.device).eval()
    # 获取图片总数
    file_list = os.listdir(config.deepspace.original_video_img)
    img_num = len(file_list)

    for i in tqdm(range(img_num), desc="真实图片转换成动漫图片"):
        f_name = str(i + 1) + '.jpg'
        item = os.path.join(config.deepspace.original_video_img, f_name)
        images = cv2.imread(item)
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        images = images.astype(np.float32) / 127.5 - 1.0
        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(config.deepspace.device)
        with torch.no_grad():
            result = model(images, False)  # 转换动漫风格
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = ((result + 1.) * 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(config.deepspace.anime_video_img + str(i) + ".jpg", result)
    return


def anime_to_high_resolution():
    """把低分辨率的动漫图片转换成高分辨率的动漫图片
    """
    # 获取图片总数
    file_list = os.listdir(config.deepspace.anime_video_img)
    img_num = len(file_list)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=config.deepspace.model_path,
        model=model,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        half=False)

    for i in tqdm(range(img_num), desc="低分辨率的动漫图片转换成高分辨率"):
        f_name = str(i + 1) + '.jpg'
        item = os.path.join(config.deepspace.anime_video_img, f_name)
        images = cv2.imread(item)
        with torch.no_grad():
            try:
                result, _ = upsampler.enhance(images, outscale=config.deepspace.scale)
                cv2.imwrite(config.deepspace.high_video_img + str(i) + ".jpg", result)
            except:
                continue
    return


# 把图片转动漫并合成视频
def ani2video():
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=config.deepspace.pretrained, device=config.deepspace.device,)
    model.to(config.deepspace.device).eval()
    # 获取图片总数
    file_list = os.listdir(config.deepspace.original_video_img)
    img_num = len(file_list)

    # 查看原始视频的参数
    cap = cv2.VideoCapture(config.deepspace.original_video)
    ret, frame = cap.read()
    # 任选一张图片查看高度和宽度
    # result = model.style_transfer(images=[cv2.imread(os.path.join(config.deepspace.original_video_img, file_list[0]))])
    image = cv2.imread(os.path.join(config.deepspace.original_video_img, file_list[0]))
    height = image.shape[0]
    width = image.shape[1]

    config.deepspace.fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率

    # 把参数用到我们要创建的视频上
    video = cv2.VideoWriter(config.deepspace.img2video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), config.deepspace.fps, (width, height))  # 创建视频流对象
    """
    参数1 即将保存的文件路径
    参数2 VideoWriter_fourcc为视频编解码器 cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 文件名后缀为.mp4
    参数3 为帧播放速率
    参数4 (width,height)为视频帧大小
    """
    for i in tqdm(range(img_num)):
        f_name = str(i + 1) + '.jpg'
        item = os.path.join(config.deepspace.original_video_img, f_name)
        images = cv2.imread(item)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(config.deepspace.device)
        with torch.no_grad():
            result = model(images, False)  # 转换动漫风格
        result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = ((result + 1.) * 127.5).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        video.write(result)  # 把图片写进视频
    video.release()  # 释放


def sound2video():
    """
    # 从原始视频上提取声音合成到新生成的视频上

    """
    # 读取原始视频
    video_o = VideoFileClip(config.deepspace.original_video)
    # 获取原始视频的音频部分
    audio_o = video_o.audio
    # 读取新生成视频
    video_clip = VideoFileClip(config.deepspace.img2video)
    # 指向新生成视频的音频部分
    video_clip2 = video_clip.set_audio(audio_o)
    # 修改音频部分并输出最终视频
    video_clip2.write_videofile(config.deepspace.output_video)


if __name__ == '__main__':

    # 第一步：视频->图像
    Path(config.deepspace.original_video_img).mkdir(parents=True, exist_ok=True)
    video_to_image()

    # 第二步：转换为动漫效果
    Path(config.deepspace.anime_video_img).mkdir(parents=True, exist_ok=True)
    real_to_anime()

    # 第三步：转换为高分辨率图片
    Path(config.deepspace.high_video_img).mkdir(parents=True, exist_ok=True)
    anime_to_high_resolution()

    # 第四步：合成视频
    image_to_video()

    # 第五步：加上原始音频
    # if not os.path.exists(config.deepspace.output_video):
    sound2video()
    # else:
    # print('最终视频已存在，请查看输出路径')

    # 第六步：删除临时文件
    # shutil.rmtree(config.deepspace.original_video_img)
    # shutil.rmtree(config.deepspace.anime_video_img)
    # shutil.rmtree(config.deepspace.high_video_img)
    # shutil.rmtree(config.deepspace.img2video)
