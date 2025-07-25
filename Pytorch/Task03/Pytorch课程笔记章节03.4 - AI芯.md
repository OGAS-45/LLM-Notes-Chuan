---
文档名-title: Pytorch课程笔记Task03.4 - AI芯
创建时间-create time: 2025-07-18 20:29
更新时间-modefived time: 2025-07-18 20:29 星期五
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---
# AI硬件加速设备

## CPU和GPU查漏补缺

CPU 的主要职责并不只是数据运算，还需要执行存储读取、指令分析、分支跳转等命令。

深度学习算法通常需要进行海量的数据处理，用 CPU 执行算法时，CPU 将花费大量的时间在数据/指令的读取分析上，而 CPU的频率、内存的带宽等条件又不可能无限制提高，因此限制了处理器的性能。

而 GPU 的控制相对简单，大部分的晶体管可以组成各类专用电路、多条流水线，使得 GPU 的计算速度远高于CPU；同时 GPU 拥有了更加强大的浮点运算能力，可以缓解深度学习算法的训练难题，释放人工智能的潜能。

GPU没有独立工作的能力，必须由CPU进行控制调用才能工作，且GPU的功耗一般比较高。

## AI芯

高功耗低效率的GPU不再能满足AI训练的要求，为此，一大批功能相对单一，但速度更快的专用集成电路相继问世。

定制的特性有助于提高 ASIC 的性能功耗比。ASIC的缺点是电路设计需要定制，相对开发周期长，功能难以扩展。但在功耗、可靠性、集成度等方面都有优势，尤其在要求高性能、低功耗的移动应用端体现明显。

TPU即Tensor Processing Unit，中文名为张量处理器。神经网络构建一个专用的集成电路（ASIC）

https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/2.4%20AI%E7%A1%AC%E4%BB%B6%E5%8A%A0%E9%80%9F%E8%AE%BE%E5%A4%87.html#id1

Matrix Multiply Unit矩阵处理器作为TPU的核心部分，它可以在单个时钟周期内处理数十万次矩阵（Matrix）运算。MMU有着与传统CPU、GPU截然不同的架构，称为脉动阵列（systolic array）。之所以叫“脉动”，是因为在这种结构中，数据一波一波地流过芯片，与心脏跳动供血的方式类似。而如下图所示，CPU和GPU在每次运算中都需要从多个寄存器（register）中进行存取，而TPU的脉动阵列将多个运算逻辑单元（ALU）串联在一起，复用从一个寄存器中读取的结果。每个ALU单元结构简单，一般只包含乘法器、加法器以及寄存器三部分，适合大量堆砌。

![[Pasted image 20250718203331.png]]

但是，在极大增加数据复用、降低内存带宽压力的同时，脉动阵列也有两个缺点，即数据重排和规模适配。第一，脉动矩阵主要实现向量/矩阵乘法。以CNN计算为例，CNN数据进入脉动阵列需要调整好形式，并且严格遵循时钟节拍和空间顺序输入。数据重排的额外操作增加了复杂性。第二，在数据流经整个阵列后，才能输出结果。当计算的向量中元素过少，脉动阵列规模过大时，不仅难以将阵列中的每个单元都利用起来，数据的导入和导出延时也随着尺寸扩大而增加，降低了计算效率。因此在确定脉动阵列的规模时，在考虑面积、能耗、峰值计算能力的同时，还要考虑典型应用下的效率。

## 技术特点

TPU的架构属于Domain-specific Architecture，也就是特定领域架构。它的定位准确，架构简单，单线程控制，定制指令集使得它在深度学习运算方面效率极高，且容易扩展。

TPU这种特点也决定它只能被限制用于深度学习加速场景。

传统GPU由于片上内存较少，因此在运行过程中需要不断地去访问片外动态随机存取存储器（DRAM），从而在一定程度上浪费了不必要的能耗。与CPU和GPU相比，TPU的控制单元更小，更容易设计，面积只占了整个冲模的2%，给片上存储器和运算单元留下了更大的空间。如上图所示的TPU一代架构中，总共设计了占总芯片面积35%的内存，其中包括24MB的局部内存、4MB的累加器内存，以及用于与主控处理器对接的内存。这一比例大大超出了GPU等通用处理器，节约了大量片外数据访存能耗，使得TPU计算的能效比大大提高。从TPU二代开始采用HBM片上高带宽内存，虽然和最新一代GPU片上内存技术相同，但是TPU芯片的面积要远远小于GPU。硅片越小，成本越低，良品率也越高。


## NPU

NPU即Neural-network Processing Unit，中文名为神经网络处理器，它采用“数据驱动并行计算”的架构，特别擅长处理视频、图像类的海量多媒体数据。

从技术角度看，深度学习实际上是一类多层大规模人工神经网络。它模仿生物神经网络而构建，由若干人工神经元结点互联而成。神经元之间通过突触两两连接，突触记录了神经元间联系的权值强弱。由于深度学习的基本操作是神经元和突触的处理，神经网络中存储和处理是一体化的，都是通过突触权重来体现，而冯·诺伊曼结构中，存储和处理是分离的，分别由存储器和运算器来实现，二者之间存在巨大的差异。当用现有的基于冯·诺伊曼结构的经典计算机(如X86处理器和英伟达GPU)运行神经网络应用时，就不可避免地受到存储和处理分离式结构的制约，因而影响效率。因此，专门针对人工智能的专业芯片NPU更有研发的必要和需求。

在NPU的设计上，中国走在了世界前列。下面我们将以寒武纪的DianNao系列架构为例，来简要介绍NPU。


如图所示，上图中浅蓝色的部分就是用硬件逻辑模拟的神经网络架构，称为NFU（Neural Functional Units）。它可以被细分为三个部分，即途中的NFU-1，NFU-2，和NFU-3。

NFU-1是乘法单元，它采用16bit定点数乘法器，1位符号位，5位整数位，10位小数位。该部分总计有256个乘法器。这些乘法器的计算是同时的，也就是说，在一个周期内可以执行256次乘法。

NFU-2是加法树，总计16个，每一个加法树都是8-4-2-1这样的组成结构，即就是每一个加法树中都有15个加法器。

NFU-3是非线性激活函数，该部分由分段线性近似实现非线性函数，根据前面两个单元计算得到的刺激量，从而判断是否需要激活操作。

当需要实现向量相乘和卷积运算时，使用NFU-1完成对应位置元素相乘，NFU-2完成相乘结果相加，最后由NFU-3完成激活函数映射。完成池化运算时，使用NFU-2完成多个元素取最大值或取平均值运算。由此分析，尽管该运算模块非常简单，也覆盖了神经网络所需要的大部分运算。

### DianNao
https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/2.4%20AI%E7%A1%AC%E4%BB%B6%E5%8A%A0%E9%80%9F%E8%AE%BE%E5%A4%87.html#diannao
### DaDianNao
### ShiDianNao
### PuDianNao