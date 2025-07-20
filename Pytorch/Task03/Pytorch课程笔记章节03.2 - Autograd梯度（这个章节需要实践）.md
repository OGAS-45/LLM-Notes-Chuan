---
文档名-title: Pytorch课程笔记Task03.2 - Autograd梯度
创建时间-create time: 2025-07-18 11:53
更新时间-modefived time: 2025-07-18 11:53 星期五
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

# Autograd

PyTorch 中，所有神经网络的核心是 `autograd` 包。autograd包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义 ( define-by-run ）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。

==`torch.Tensor` 是这个包的核心类。如果设置它的属性 `.requires_grad` 为 `True`，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用 `.backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到`.grad`属性。==

PyTorch 的 `autograd` 模块是自动微分的核心。当你对张量进行操作时，PyTorch 会记录这些操作的计算图，以便在反向传播时计算梯度。每个操作都会生成一个对应的“反向传播节点”，这些节点在计算图中用于存储梯度计算所需的中间信息。

### 2. **`PowBackward0` 是什么？**

`PowBackward0` 是 PyTorch 中的一个内部类，用于表示幂运算（如 `x.pow(y)` 或 `x ** y`）的反向传播节点。当你对一个张量进行幂运算时，PyTorch 会创建一个 `PowBackward0` 对象，用于在反向传播时计算该操作的梯度。

### 3. **为什么看到这个输出？**

通常情况下，你不会直接看到 `PowBackward0` 对象的输出，除非你在调试过程中不小心打印了某个反向传播节点。例如：

Python

复制

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # 这里会创建一个 PowBackward0 节点
print(y.grad_fn)  # 打印反向传播节点
```

输出可能是：

`<PowBackward0 object at 0x000001CB45988C70>`

这里的 `y.grad_fn` 是一个反向传播节点对象，表示 `y` 的梯度计算依赖于 `x ** 2` 这个操作。


还有一个类对于`autograd`的实现非常重要：`Function`。`Tensor` 和 `Function` 互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史。每个张量都有一个`.grad_fn`属性，该属性引用了创建 `Tensor` 自身的`Function`(除非这个张量是用户手动创建的，即这个张量的`grad_fn`是 `None` )。



## 计算导数

如果需要计算导数，可以在 `Tensor` 上调用 `.backward()`。如果 `Tensor` 是一个标量(即它包含一个元素的数据），则不需要为 `backward()` 指定任何参数，但是如果它有更多的元素，则需要指定一个`gradient`参数，该参数是形状匹配的张量。


# 梯度

因为 `out` 是一个标量，因此`out.backward()`和 `out.backward(torch.tensor(1.))` 等价。

out.backward()会输出导数 `d(out)/dx`
