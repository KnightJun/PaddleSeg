简体中文 | [English](README.md)

# PaddleSeg（动态图版本）

本目录提供了PaddleSeg的动态图版本，目前已经完成了模型训练、评估、数据处理等功能，在未来的版本中，PaddleSeg将会启动默认的动态图模式。目前该目录处于实验阶段，如果您在使用过程中遇到任何问题，请通过issue反馈给我们，我们将会在第一时间跟进处理。

## 模型库

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann)|✔|✔|||
|[BiSeNetv2](./configs/bisenet)|-|-|-|-|
|[DANet](./configs/danet)|✔|✔|||
|[Deeplabv3](./configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](./configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](./configs/fastscnn)|-|-|-|-|
|[FCN](./configs/fcn)|||✔|✔|
|[GCNet](./configs/gcnet)|✔|✔|||
|[GSCNN](./configs/gscnn)|✔|✔|||
|[HarDNet](./configs/hardnet)|-|-|-|-|
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|[PSPNet](./configs/pspnet)|✔|✔|||
|[U-Net](./configs/unet)|-|-|-|-|
|[U<sup>2</sup>-Net](./configs/u2net)|-|-|-|-|
|[Att U-Net](./configs/attention_unet)|-|-|-|-|
|[U-Net++](./configs/unet_plusplus)|-|-|-|-|

## 数据集

- [x] Cityscapes
- [x] Pascal VOC
- [x] ADE20K
- [ ] Pascal Context
- [ ] COCO stuff

## 安装

1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0rc

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。


安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。


2. 下载PaddleSeg代码
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：


```shell
cd PaddleSeg
export PYTHONPATH=`pwd`
# windows下请执行以下命令
# set PYTHONPATH=%cd%
pip install -r requirements.txt
```

## 训练
```shell
python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [API使用教程](https://aistudio.baidu.com/aistudio/projectdetail/1339458)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./configs/)
* [Add New Components](./docs/add_new_model.md)

## 代码贡献

* 非常感谢[jm12138](https://github.com/jm12138)贡献U<sup>2</sup>-Net模型。
* 非常感谢[zjhellofss](https://github.com/zjhellofss)（傅莘莘）贡献Attention U-Net模型，和Dice loss损失函数。
* 非常感谢[liuguoyu666](https://github.com/liguoyu666)贡献U-Net++模型。