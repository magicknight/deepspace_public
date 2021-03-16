#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
echo "---------------------------------"
for i in {61..100..1}
  do 
     echo "Downloading data playground-${i}.h5"
     wget -c http://hep.tsinghua.edu.cn/~orv/dc/2021/playground-${i}.h5  -P data/tsinghua > data/tsinghua/${i}.log 2>&1 &
     echo "---------------------------------"
 done
