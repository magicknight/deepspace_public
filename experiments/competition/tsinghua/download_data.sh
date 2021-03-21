#!/bin/bash

if [ $1 == "playground" ]; then
  url='http://hep.tsinghua.edu.cn/~orv/dc/2021/playground-'
else
  url='http://hep.tsinghua.edu.cn/~orv/dc/2021/pre-'
fi

echo "Bash version ${BASH_VERSION}..."
echo "---------------------------------"
for i in {1..9..1}
  do 
     echo "Downloading data ${url}${i}.h5"
     wget -c ${url}${i}.h5  -P data/tsinghua/first_round > data/tsinghua/first_round/${i}.log 2>&1 &
     echo "---------------------------------"
 done
