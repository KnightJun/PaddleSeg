[English](README.md) | 简体中文
# PaddleSeg RKNPU2 C++部署示例

本目录下用于展示PaddleSeg系列模型在RKNPU2上的部署，以下的部署过程以PPHumanSeg为例子。

## 1. 部署环境准备 
在部署前，需确认以下两个步骤:

1. 软硬件环境满足要求
2. 根据开发环境，下载预编译部署库或者从头编译FastDeploy仓库

以上步骤请参考[RK2代NPU部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/rknpu2/rknpu2.md)实现

## 2. 部署模型准备

模型转换代码请参考[模型转换文档](../README.md)

## 3. 运行部署示例

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleSeg.git 
cd PaddleSeg/deploy/fastdeploy/semantic_segmentation/rockchip/rknpu2/cpp

# 编译部署示例
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j8

wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip -qo images.zip

# 运行部署示例
./infer_demo model/Portrait_PP_HumanSegV2_Lite_256x144_infer/ images/portrait_heng.jpg
```

## 4. 更多指南
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisableNormalizeAndPermute(C++)或`disable_normalize_and_permute(Python)，在预处理阶段禁用归一化以及数据格式的转换。

- [模型介绍](../../)
- [Python部署](../python)
- [转换PPSeg RKNN模型文档](../README.md)