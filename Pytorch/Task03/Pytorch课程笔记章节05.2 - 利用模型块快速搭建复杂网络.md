---
文档名-title: Pytorch课程笔记章节05.2 - 利用模型块快速搭建复杂网络
创建时间-create time: 2025-07-23 22:58
更新时间-modefived time: 2025-07-23 22:58 星期三
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

上一节中我们介绍了怎样定义PyTorch的模型，其中给出的示例都是用`torch.nn`中的层来完成的。这种定义方式易于理解，在实际场景下不一定利于使用。当模型的深度非常大时候，使用`Sequential`定义模型结构需要向其中添加几百行代码，使用起来不甚方便。

对于大部分模型结构（比如ResNet、DenseNet等），我们仔细观察就会发现，虽然模型有很多层， 但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，这样将会极大便利模型构建的过程。

## 5.2.1 U-Net简介

U-Net是分割 (Segmentation) 模型的杰作，在以医学影像为代表的诸多领域有着广泛的应用。U-Net模型结构如下图所示，通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展。

![[Pasted image 20250723225900.png]]

## 5.2.2 U-Net模型块分析 - 上下采样 池化和非池

结合上图，不难发现U-Net模型具有非常好的对称性。模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接；同时位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”。此外还有输入和输出处理等其他组成部分。由于模型的形状非常像英文字母的“U”，因此被命名为“U-Net”。

组成U-Net的模型块主要有如下几个部分：

- 每个子块内部的两次卷积（Double Convolution）
    
- 左侧模型块之间的下采样连接，即最大池化（Max pooling）
    
- 右侧模型块之间的上采样连接（Up sampling）
    
- 输出层的处理


## 5.2.3 U-Net模型块实现

在使用PyTorch实现U-Net模型时，我们不必把每一层按序排列显式写出，这样太麻烦且不宜读，一种比较好的方法是先定义好模型块，再定义模型块之间的连接顺序和计算方式。就好比装配零件一样，我们先装配好一些基础的部件，之后再用这些可以复用的部件得到整个装配体。

## 5.2.4 利用模型块组装U-Net

```
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

# 5.3 PyTorch修改模型

除了自己构建PyTorch模型外，还有另一种应用场景：我们已经有一个现成的模型，但该模型中的部分结构不符合我们的要求，为了使用模型，我们需要对模型结构进行必要的修改。随着深度学习的发展和PyTorch越来越广泛的使用，有越来越多的开源模型可以供我们使用，很多时候我们也不必从头开始构建模型。因此，掌握如何修改PyTorch模型就显得尤为重要。

## 5.3.1 修改模型层

这里模型结构是为了适配ImageNet预训练的权重，因此最后全连接层（fc）的输出节点数是1000。

假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层，将其输出节点数替换为10。另外，我们觉得一层全连接层可能太少了，想再加一层。可以做如下修改：

```
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
    
net.fc = classifier
```

这里的操作相当于将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构，该结构是我们自己定义的。这里使用了第一节介绍的Sequential+OrderedDict的模型定义方式。至此，我们就完成了模型的修改，现在的模型就可以去做10分类任务了。

## 5.3.2 添加外部输入

有时候在模型训练中，除了已有模型的输入之外，还需要输入额外的信息。比如在CNN网络中，我们除了输入图像，还需要同时输入图像对应的其他信息，这时候就需要在已有的CNN网络中添加额外的输入变量。基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改。

我们以torchvision的resnet50模型为基础，任务还是10分类任务。不同点在于，我们希望利用已有的模型结构，在倒数第二层增加一个额外的输入变量add_variable来辅助预测。

这里的实现要点是通过torch.cat实现了tensor的拼接。torchvision中的resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），先将1000维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，最后通过全连接层映射到指定的输出维度10。

另外这里对外部输入变量"add_variable"进行unsqueeze操作是为了和net输出的tensor保持维度一致，常用于add_variable是单一数值 (scalar) 的情况，此时add_variable的维度是 (batch_size, )，需要在第二维补充维数1，从而可以和tensor进行torch.cat操作。对于unsqueeze操作可以复习下2.1节的内容和配套代码。

## 5.3.3 添加额外输出

有时候在模型训练中，除了模型最后的输出外，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。基本的思路是修改模型定义中forward函数的return变量。