---
文档名-title: Pytorch课程笔记章节04.1 -
创建时间-create time: 2025-07-23 21:00
更新时间-modefived time: 2025-07-23 21:00 星期三
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---
## 5.1.2 Sequential
当模型的前向计算为简单串联各个层的计算时， `Sequential` 类可以通过更加简单的方式定义模型。它可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 `Module` 的实例，⽽模型的前向计算就是将这些实例按添加的顺序逐⼀计算。

## 5.1.3 ModuleList

对应模块为`nn.ModuleList()`。

`ModuleList` 接收一个子模块（或层，需属于`nn.Module`类）的列表作为输入，然后也可以类似`List`那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。

要特别注意的是，`nn.ModuleList` 并没有定义一个网络，它只是将不同的模块储存在一起。`ModuleList`中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用for循环即可完成：

## 5.1.4 ModuleDict

## 5.1.5 三种方法的比较与适用场景

`Sequential`适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写`__init__`和`forward`；

ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。