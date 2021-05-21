#!/bin/bash

if [ $1 == "playground" ]; then
  url='http://hep.tsinghua.edu.cn/~orv/dc/2021/playground-'
fi
if [ $1 == "final" ]; then
  url='http://hep.tsinghua.edu.cn/~orv/dc/2021/final-'
fi
if [ $1 == "first" ]; then
  url='http://hep.tsinghua.edu.cn/~orv/dc/2021/pre-'
fi

echo "Bash version ${BASH_VERSION}..."
echo "---------------------------------"
echo $url
for i in {2..20..1}
  do 
     echo "Downloading data ${url}${i}.h5"
     wget -c ${url}${i}.h5  -P data/tsinghua/final > data/tsinghua/logs/${i}.log 2>&1 &
     echo "---------------------------------"
 done
