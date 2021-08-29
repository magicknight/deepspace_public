#!/usr/bin/env sh
###
 #           佛曰:  
 #                   写字楼里写字间，写字间里程序员；  
 #                   程序人员写程序，又拿程序换酒钱。  
 #                   酒醒只在网上坐，酒醉还来网下眠；  
 #                   酒醉酒醒日复日，网上网下年复年。  
 #                   但愿老死电脑间，不愿鞠躬老板前；  
 #                   奔驰宝马贵者趣，公交自行程序员。  
 #                   别人笑我忒疯癫，我笑自己命太贱；  
 #                   不见满街漂亮妹，哪个归得程序员？
 # 
 # @Description: 
 # @Author: Zhihua Liang
 # @Github: https://github.com/magicknight
 # @Date: 2021-08-25 13:40:19
 # @LastEditors: Zhihua Liang
 # @LastEditTime: 2021-08-27 17:28:32
 # @FilePath: /home/zhihua/framework/deepspace/scripts/data_preprocess/shapenet.sh
###

# This script downloads the ShapeNet data and unzips it.

# do not change this name
# dataset_dir="/home/zhihua/gcs/deepspace/public/shapenet/core_v1"
dataset_dir="/home/zhihua/data/shapenetcore"
zip_file="ShapeNetCore.v1.zip"

# if you have already had the same version of dataset, you can 
# create soft link like this:
# >> ln -s <path/to/ShapeNetCore/> shapenetcore

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo "DIR: $DIR"
cd $DIR

if [ -f $zip_file ];
then
  echo "Good. you already have the zip file downloaded."
else
  echo "Please visit http://shapenet.cs.stanford.edu to request ShapeNet data and then put the zip file in this folder and then run this script again.."
fi

echo "Unzipping..."

mkdir $dataset_dir
unzip ShapeNetCore.v1.zip && rm -f ShapeNetCore.v1.zip
mv ShapeNetCore.v1/* $dataset_dir && rm -rf ShapeNetCore.v1
cd $dataset_dir
for zipfile in `ls *.zip`; do unzip $zipfile; rm $zipfile; done
cd ..

echo "Done."