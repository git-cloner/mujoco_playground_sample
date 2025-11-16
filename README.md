# Mujoco Playground实践

实践Mujoco Playground。包括安装以及对dm_control_suite、locomotion的测试。

![](https://gitclone.com/download1/playground/sample0.gif) ![](https://gitclone.com/download1/playground/sample1.gif)

## 一、环境要求

GPU：至少24G显存，因为训练时需要18G左右显存。如RTX4090、RTX3090等

操作系统：Linux，如Ubuntu22.04

基础软件：NVIDIA显卡驱动、CUDA12.4、Anaconda（Python虚拟环境）

## 二、环境安装

### 1、安装ffmpeg

```shell
# 更新系统
sudo apt update
# 安装ffmpeg
sudo apt install -y ffmpeg
```

### 2、创建虚拟环境

```shell
# 建立虚拟环境
conda create -n mujoco python==3.12 -y
# 激活虚拟环境
conda activate mujoco
```

### 3、安装依赖库

```shell
# clone源码
git clone https://github.com/git-cloner/mujoco_playground_sample
# 切换目录
cd mujoco_playground_sample
# 安装依赖库
pip install -r requirements.txt \
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 三、运行测试程序

```shell
# 解决“untimeError: Unable to load cuSPARSE. Is it installed?”问题
unset LD_LIBRARY_PATH
# 测试dm_control_suite
CUDA_VISIBLE_DEVICES=0 python dm_control_suite.py
# 测试运动
CUDA_VISIBLE_DEVICES=0 python locomotion.py
```

## 四、参考

```shell
# 官网例子
git clone https://github.com/google-deepmind/mujoco_playground
cd mujoco_playground/learning/notebooks
# jupyter 转 python
jupyter nbconvert --to python dm_control_suite.ipynb
# 其他参考文件
https://research.mels.ai/ide?mels=UnitreeGo1.qkazy
```




