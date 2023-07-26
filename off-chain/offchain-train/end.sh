#!/bin/bash

process_name=$1

python test.py

ps -ef | grep ${process_name} | grep -v grep | awk '{print $2}' | xargs kill -9