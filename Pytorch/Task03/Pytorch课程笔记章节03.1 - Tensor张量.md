---
文档名-title: Pytorch课程笔记Task03.1 - PyTorch知识点
创建时间-create time: 2025-07-18 11:16
更新时间-modefived time: 2025-07-18 11:16 星期五
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

# 张量Tensor - 数据容器

多数情况下，它包含数字，有时候它也包含字符串，但这种情况比较少。

|张量维度|代表含义|
|---|---|
|0维张量|代表的是标量（数字）|
|1维张量|代表的是向量|
|2维张量|代表的是矩阵|
|3维张量|时间序列数据 股价 文本数据 单张彩色图片(**RGB**)|
|3维|时间序列|
|4维|图像|
|5维|视频|

例子：一个图像可以用三个字段表示：
(width, height, channel) = 3D
> 为什么只有3维，这样肯定不足以描述一个图像以及里面的信息

在PyTorch中， `torch.Tensor` 是存储和变换数据的主要工具。如果你之前用过`NumPy`，你会发现 `Tensor` 和NumPy的多维数组非常类似。然而，`Tensor` 提供GPU计算和自动求梯度等更多功能，这些使 `Tensor` 这一数据类型更加适合深度学习。



这个是我从这篇文章中提取出来的部分方法，分成了torch.的从零构造和x.的已有Tensor，你能帮我整理、全面细节地完善以及拓展一下我想要的常用方法和细节吗？还有最后请整理得逻辑清晰、结构明确的样子，我会归纳起来。

## 常见的从零构造Tensor的方法：

| 函数                                  | 功能                                | 示例代码                              |
| ----------------------------------- | --------------------------------- | --------------------------------- |
| `torch.rand(sizes)`                 | 生成 \[0, 1) 区间内的均匀分布随机数            | `torch.rand(4, 3)`                |
| `torch.randn(sizes)`                | 生成服从 N(0, 1) 的正态分布随机数             | `torch.randn(4, 3)`               |
| `torch.zeros(sizes)`                | 生成全 0 张量                          | `torch.zeros(4, 3)`               |
| `torch.ones(sizes)`                 | 生成全 1 张量                          | `torch.ones(4, 3)`                |
| `torch.eye(n)`                      | 生成 n×n 的单位矩阵                      | `torch.eye(3)`                    |
| `torch.arange(start, end, step)`    | 生成从 start 到 end 的等差序列，步长为 step    | `torch.arange(1, 10, 2)`          |
| `torch.linspace(start, end, steps)` | 生成从 start 到 end 的等间隔序列，共 steps 个值 | `torch.linspace(1, 10, 5)`        |
| `torch.normal(mean, std)`           | 生成均值为 mean、标准差为 std 的正态分布随机数      | `torch.normal(0, 1, size=(4, 3))` |
| `torch.randperm(n)`                 | 生成 0 到 n-1 的随机排列                  | `torch.randperm(10)`              |
| `torch.tensor(data)`                | 从数据直接构造张量                         | `torch.tensor([5.5, 3])`          |


## 常见的基于已有张量构造张量的方法：
| 方法                    | 功能                             | 示例代码                  |
| --------------------- | ------------------------------ | --------------------- |
| `x.new_ones(sizes)`   | 创建一个与 x 同设备、同数据类型的全 1 张量       | `x.new_ones(4, 3)`    |
| `x.new_zeros(sizes)`  | 创建一个与 x 同设备、同数据类型的全 0 张量       | `x.new_zeros(4, 3)`   |
| `torch.ones_like(x)`  | 创建一个与 x 同形状、同设备、同数据类型的全 1 张量   | `torch.ones_like(x)`  |
| `torch.zeros_like(x)` | 创建一个与 x 同形状、同设备、同数据类型的全 0 张量   | `torch.zeros_like(x)` |
| `torch.randn_like(x)` | 创建一个与 x 同形状、同设备、同数据类型的正态分布随机张量 | `torch.randn_like(x)` |


##  张量操作

### 3.1 基本运算

张量支持多种基本运算，包括加法、减法、乘法、除法等。

|运算|示例代码|
|:--|:--|
|加法|`x + y` 或 `torch.add(x, y)` 或 `x.add_(y)`|
|减法|`x - y` 或 `torch.sub(x, y)` 或 `x.sub_(y)`|
|逐元素乘法|`x * y` 或 `torch.mul(x, y)` 或 `x.mul_(y)`|
|逐元素除法|`x / y` 或 `torch.div(x, y)` 或 `x.div_(y)`|


### 3.2 索引与切片

张量支持类似于 NumPy 的索引和切片操作。需要注意的是，索引操作返回的张量与原张量共享内存，修改一个会影响另一个。

```python
x = torch.rand(4, 3)
# 取第二列
print(x[:, 1])
# 取第一行
print(x[0, :])
```

### 3.3 维度变换

维度变换是张量操作中常用的功能，主要有以下两种方法：

- **`torch.view()`**：改变张量的形状，但返回的新张量与原张量共享内存。
    
    ```python
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # -1 表示自动计算该维度的大小
    ```
    
- **`torch.reshape()`**：与 `torch.view()` 类似，但返回的张量可能不共享内存（取决于是否满足内存连续性条件）。
    
    ```python
    x = torch.randn(4, 4)
    y = torch.reshape(x, (16,))
    z = torch.reshape(x, (-1, 8))
    ```
    

### 3.4 广播机制

当对两个形状不同的张量进行按元素运算时，PyTorch 会触发广播机制，通过适当复制元素使两个张量形状相同后再进行运算。


```python
x = torch.arange(1, 3).view(1, 2)  # 形状为 [1, 2]
y = torch.arange(1, 4).view(3, 1)  # 形状为 [3, 1]
print(x + y)  # 广播后形状为 [3, 2]
```

### 3.5 其他操作

- **取值操作**：如果张量只有一个元素，可以使用 `.item()` 获取其值。
    
    
    ```python
    x = torch.randn(1)
    print(x.item())  # 获取单个元素的值
    ```
    
- **转置操作**：`torch.transpose(x, dim0, dim1)` 或 `x.t()`。
    
    
    ```python
    x = torch.randn(3, 4)
    y = torch.transpose(x, 0, 1)  # 交换第 0 维和第 1 维
    z = x.t()  # 等价于 transpose(0, 1)
    ```
    
- **拼接操作**：`torch.cat(tensors, dim)`。
    
    
    ```python
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = torch.cat((x, y), dim=0)  # 按第 0 维拼接
    ```