#! /bin/bash
# 用来执行脚本
# $1 需要使用的GPU数量
# $2 需要运行的脚本文件

var1=0
var2=0
array=()
for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  do
    if [ "$i" -le 1000 ]
    then
      array[$var1]=$var2
      var1=$(expr $var1 + 1 )
    fi

    if [ "$var1" -ge "$1" ]
    then
       break
    fi
    var2=$(expr $var2 + 1)
  done

if [ "$var1" -ge "$1" ]
then
#  echo ${array[*]}
  export CUDA_VISIBLE_DEVICES="${array[*]}"
  sh "$2"
else
  echo "NO free GPU!"
fi




