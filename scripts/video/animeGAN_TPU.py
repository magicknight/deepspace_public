#!/usr/bin/env python3


import numpy as np
import torch

from moviepy.editor import *
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from imageio import imread, imwrite
import shutil

from basicsr.archs.rrdbnet_arch import RRDBNet

from imagedataset import get_dataloader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from commontools.setup import config, logger

# pprint(config)
# Config
# 原始视频地址
# config.deepspace.original_video = '../data/高中教师.mp4'
# # 提取视频图像的存放地址
# config.deepspace.original_video_img = '../image/高中教师_img/'
# # 合成视频存放地址
# # 添加声音后的视频最终输出地址
# config.deepspace.output_video = './高中教师_audio.mp4'


def save_img(
    img_array, root, index=None, progress_bar=True, color_mode=None, normalize_mode=None
):
    """把 numpy array 按序列保存为图片"""
    if index is None:
        index = range(img_array.shape[0])
    if progress_bar:
        batches = tqdm(range(img_array.shape[0]), desc="保存图片")
    else:
        batches = range(img_array.shape[0])
    for i in batches:
        if normalize_mode == "tensorflow":
            result = ((img_array[i] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        elif normalize_mode == "torch":
            result = ((img_array[i]) * 255).clip(0, 255).astype(np.uint8)
        imwrite(Path(root) / (str(index[i]) + ".jpg"), result)


# 从视频提取图片
def video_to_image():
    """把视频转换成图片"""
    """
    # 截取视频中的一段时间
    """
    clip = VideoFileClip(config.deepspace.original_video)
    config.deepspace.fps = clip.fps
    if config.deepspace.end_time != 0:
        clip = clip.subclip(config.deepspace.start_time, config.deepspace.end_time)
    if "resize" in config.deepspace:
        clip = clip.resize(height=config.deepspace.resize)
    clip.write_images_sequence(
        config.deepspace.original_video_img + "%03d.jpg", fps=config.deepspace.fps
    )
    return


def image_to_video():
    """把高分辨率动漫图片转换成视频"""
    # 读取原始视频
    video_o = VideoFileClip(config.deepspace.original_video)
    # 获取原始视频的音频部分
    audio_o = video_o.audio
    # 获取图片
    images = sorted(
        list(Path(config.deepspace.high_video_img).glob("*.jpg")),
        key=lambda x: int(x.stem),
    )
    images = [str(i) for i in images]
    video_clip = ImageSequenceClip(images, fps=config.deepspace.fps)
    # 指向新生成视频的音频部分
    video_clip = video_clip.set_audio(audio_o)
    # 修改音频部分并输出最终视频
    video_clip.write_videofile(config.deepspace.output_video)
    return


@torch.no_grad()
def real_to_anime(
    index,
):
    """把真实图片转换成动漫图片"""
    # Sets a common random seed - both for initialization and ensuring graph is the same
    torch.manual_seed(config.deepspace.seed)
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()
    logger.info("Process", index, "is using", xm.xla_real_devices([str(device)])[0])
    # Downloads model
    # Note: master goes first and downloads the model only once (xm.rendezvous)
    #   all the other workers wait for the master to be done downloading.
    # if not xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=config.deepspace.pretrained,
    )
    # if xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')

    model.to(device).eval()

    # 转换后的图片存在numpy array中
    # get data loader
    data_loader = get_dataloader(
        Path(config.deepspace.original_video_img),
        batch=config.deepspace.anime_batch,
        normalize_mode="tensorflow",
    )

    # 开始转换
    for images, index in tqdm(data_loader, desc="真实图片转换成动漫图片"):
        # 获取图片
        images = images.to(device)
        # 转换图片
        try:
            result = model(images, False)  # 转换动漫风格
        except Exception as e:
            print(e)
            continue
        # 保存图片
        save_img(
            result.permute(0, 2, 3, 1).cpu().numpy(),
            config.deepspace.anime_video_img,
            index=index.cpu().numpy(),
            progress_bar=False,
            color_mode="RGB",
            normalize_mode="tensorflow",
        )
    xm.rendezvous("finish_real_to_anime")
    return


@torch.no_grad()
def anime_to_high_resolution(index):
    """把低分辨率的动漫图片转换成高分辨率的动漫图片"""
    # Sets a common random seed - both for initialization and ensuring graph is the same
    torch.manual_seed(config.deepspace.seed)
    # Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()

    # Downloads model
    # 动画模型
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=config.deepspace.scale,
    )
    # 通用模型
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=config.deepspace.scale,)
    model.load_state_dict(
        torch.load(config.deepspace.model_path)["params_ema"], strict=True
    )
    model.to(device).eval()

    # 转换后的图片存在numpy array中
    result_array = []
    # get data loader
    data_loader = get_dataloader(
        Path(config.deepspace.anime_video_img),
        batch=config.deepspace.hs_batch,
        normalize_mode="torch",
    )
    # 开始转换
    for images, index in tqdm(data_loader, desc="低分辨率的动漫图片转换成高分辨率"):
        # 获取图片
        images = images.to(device)
        # 转换图片
        try:
            result = model(images)
        except Exception as e:
            print(e)
            continue
        # 保存图片
        save_img(
            result.permute(0, 2, 3, 1).cpu().numpy(),
            config.deepspace.high_video_img,
            index=index.cpu().numpy(),
            progress_bar=False,
            normalize_mode="torch",
        )
        # result_array.append(result.permute(0, 2, 3, 1).cpu().numpy())
    # list to numpy array
    # result_array = np.concatenate(result_array, axis=0)
    # 保存图片
    # save_img(result_array, config.deepspace.high_video_img)
    # xm.rendezvous("finish_anime_to_high_resolution")
    xm.rendezvous(config.deepspace.original_video)
    return


def cut_video():
    """
    # 截取视频中的一段时间
    """
    video_path = Path(config.deepspace.original_video)
    video_clip = VideoFileClip(config.deepspace.original_video)
    video_clip.subclip(
        config.deepspace.start_time, config.deepspace.end_time
    ).write_videofile(str(video_path.parent / (video_path.stem + "_cut.mp4")))
    # config.deepspace.original_video = str(video_path.parent / (video_path.stem + '_cut.mp4'))


def process_video():
    """process video"""
    # 第一步：视频->图像
    Path(config.deepspace.original_video_img).mkdir(parents=True, exist_ok=True)
    video_to_image()
    logger.info("视频->图像完成")

    # 第二步：转换为动漫效果
    Path(config.deepspace.anime_video_img).mkdir(parents=True, exist_ok=True)
    if config.deepspace.anime:
        xmp.spawn(real_to_anime, args=(), nprocs=8, start_method="fork")
    else:
        config.deepspace.anime_video_img = config.deepspace.original_video_img
    logger.info("转换为动漫效果完成")

    # 第三步：转换为高分辨率图片
    if config.deepspace.high_resolution and config.deepspace.scale > 1:
        Path(config.deepspace.high_video_img).mkdir(parents=True, exist_ok=True)
        xmp.spawn(anime_to_high_resolution, args=(), nprocs=8, start_method="fork")
    else:
        config.deepspace.high_video_img = config.deepspace.anime_video_img
    logger.info("转换为高分辨率图片完成")

    # 第四步：合成视频
    image_to_video()
    logger.info("合成视频完成")

    # 第五步：删除临时文件
    shutil.rmtree(Path(config.deepspace.original_video_img))
    if config.deepspace.anime:
        shutil.rmtree(Path(config.deepspace.anime_video_img))
    shutil.rmtree(Path(config.deepspace.high_video_img))
    logger.info("删除临时文件完成")


if __name__ == "__main__":
    # get all the videos from the input directory
    video_paths = Path(config.deepspace.input_path).glob("*")
    batch = tqdm(video_paths)
    for video_path in batch:
        logger.info(f"处理视频：{video_path}")
        # change tqdm to the video name
        batch.set_description_str(video_path.stem)
        config.deepspace.original_video = str(video_path)
        config.deepspace.output_video = str(
            Path(config.deepspace.output_path) / (video_path.stem + "_out.mp4")
        )
        process_video()
