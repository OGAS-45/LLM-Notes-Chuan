---
文档名-title: Pytorch课程笔记Task03.3 - 并行计算
创建时间-create time: 2025-07-18 20:17
更新时间-modefived time: 2025-07-18 20:17 星期五
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---


## 为什么要做并行计算

PyTorch可以在编写完模型之后，让多个GPU来参与训练，减少训练时间。

## 为什么需要CUDA

`CUDA`是NVIDIA提供的一种GPU并行计算框架。对于GPU本身的编程，使用的是`CUDA`语言来实现的。但是，在我们使用PyTorch编写深度学习代码时，使用的`CUDA`又是另一个意思。在PyTorch使用 `CUDA`表示要开始要求我们的模型或者数据开始使用GPU了。

在编写程序中，当我们使用了 `.cuda()` 时，其功能是让我们的模型或者数据从CPU迁移到GPU上（默认是0号GPU）当中，通过GPU开始计算。

注：

1. 我们使用GPU时使用的是`.cuda()`而不是使用`.gpu()`。这是因为当前GPU的编程接口采用CUDA，但是市面上的GPU并不是都支持CUDA，只有部分NVIDIA的GPU才支持，AMD的GPU编程接口采用的是OpenCL，在现阶段PyTorch并不支持。
    
2. 数据在GPU和CPU之间进行传递时会比较耗时，我们应当尽量避免数据的切换。
    
3. GPU运算很快，但是在使用简单的操作时，我们应该尽量使用CPU去完成。

## 当服务器上有多个GPU

1. 当我们的服务器上有多个GPU，我们应该指明我们使用的GPU是哪一块，如果我们不设置的话，tensor.cuda()方法会默认将tensor保存到第一块GPU上，等价于tensor.cuda(0)，这将有可能导致爆出`out of memory`的错误。我们可以通过以下两种方式继续设置。
    
```
    1.  #设置在文件最开始部分
        import os
        os.environ["CUDA_VISIBLE_DEVICE"] = "2" # 设置默认的显卡
        
    2.  CUDA_VISBLE_DEVICE=0,1 python train.py # 使用0，1两块GPU
```


## 常见的并行方法

1. 网络结构分布到不同的设备中(Network partitioning)

![[Pasted image 20250718202110.png]]

问题：不同模型组件在不同的GPU上时，GPU之间的传输就很重要，对于GPU之间的通信是一个考验。但是GPU的通信在这种密集任务中很难办到

2. 同一层的任务分布到不同数据中(Layer-wise partitioning)

![[Pasted image 20250718202215.png]]

问题：在我们需要大量的训练，同步任务加重的情况下，会出现和第一种方式一样的问题。


3. 不同的数据分布到不同的设备中，执行相同的任务(Data parallelism)

![[Pasted image 20250718202234.png]]

主流：这种方式可以解决之前模式遇到的通讯问题。现在的主流方式是**数据并行**的方式(Data parallelism)

## 使用CUDA加速训练

在PyTorch框架下，CUDA的使用变得非常简单，我们只需要显式的将数据和模型通过`.cuda()`方法转移到GPU上就可加速我们的训练。如下：
```
model = Net()
model.cuda() # 模型显示转移到CUDA上

for image,label in dataloader:
    # 图像和标签显示转移到CUDA上
    image = image.cuda() 
    label = label.cuda()
```

### 多卡情况

不写了，等需要用的时候再查。