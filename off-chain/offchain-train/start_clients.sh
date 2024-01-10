#!/bin/bash

clients=$1
dir=`date +%Y%m%d`_`date +%H:%M:%S`

mkdir ./logs/$dir
for (( i=0; i<$clients; i++ ))
do
    {
        python client0.py -client_id ${i} > ./logs/$dir/client${i}.txt 2>&1
    } &  #将上述程序块放到后台执行
done