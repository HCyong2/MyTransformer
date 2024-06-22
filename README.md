# 在图片分类任务中比较 Transformer 和 CNN

## 简介

使用了参数量相当的Transformer和传统CNN在CIFAR-10上比较其图片分类能力。

并且使用了CutMix数据增强来提升模型性能。



## 内容目录

```bash
logs                                  # 存放Tensorboard记录
models                                # 存放训练好的模型
CalParameter.py                       # 比较模型参数量
CutMix.py                             # CutMix数据增强
Train_Transformer.py                   
Train_Transformer_noCutMix.py         
```



## 使用说明

1、打开 Train_Transformer.py 修改数据集路径。

2、调整超参数进行旋转预训练。

3、使用 Tensorboard 打开 logs 文件夹查看模型训练情况

