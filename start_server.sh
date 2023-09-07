#!/bin/bash

# 启动 webui.py 并将输出重定向到 webui.log 文件
nohup python webui.py > log/webui.log 2>&1 &

# 启动 api.py 并将输出重定向到 api.log 文件
nohup python api.py > log/api.log 2>&1 &

# 在这里等待用户按下 Ctrl+C，然后停止两个进程
trap 'kill $(jobs -p)' SIGINT

# 无限循环以保持脚本运行
while true; do
    sleep 1
done
