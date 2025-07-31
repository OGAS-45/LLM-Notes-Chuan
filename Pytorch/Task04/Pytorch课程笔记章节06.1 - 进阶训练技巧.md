---
文档名-title: Pytorch课程笔记章节06.0 - 进阶训练技巧
创建时间-create time: 2025-07-31 20:30
更新时间-modefived time: 2025-07-31 20:30 星期四
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

# 深度学习实用技巧精要笔记

## 一、自定义损失函数：让模型"更懂"你的任务

### 为什么需要自定义损失函数？
- PyTorch自带的损失函数（如MSE、BCE）像"标准尺子"，适合通用任务
- 但特殊任务（如医学图像分割）需要"定制尺子"，例如：
  - **Dice Loss**：衡量两个区域重叠程度，像"比对两片拼图的契合度"
  - **Focal Loss**：专治"难学样本"，让模型更关注"顽固学生"
  - **IoU Loss**：计算"交集占并集比例"，像"评估两个气球重叠部分"

### 两种自定义方式（核心区别）
| 方式 | 适用场景 | 特点 |
|------|----------|------|
| **函数式** | 简单、一次性损失 | 像"随手写个计算器"，直接计算输出与目标的差异 |
| **类式** | 复杂、需复用的损失 | 像"定制工具箱"，继承自`nn.Module`，可保存参数和状态 |

> 💡 **关键提示**：类式更常用！因为：
> 1. 能像神经网络层一样管理内部参数
> 2. 与PyTorch自动求导系统无缝配合
> 3. 便于在不同项目中复用

### 实战经验
- **全程用PyTorch张量操作**：避免混用NumPy，否则会断开自动求导"神经"
- **常见组合技巧**：
  - Dice + BCE Loss：分割任务"黄金搭档"，兼顾形状和像素级精度
  - Focal Loss：解决类别不平衡，像"给困难样本开小灶"

---

## 二、动态调整学习率：智能调节"学习步幅"

### 为什么需要动态调整？
- 学习率太小 → 模型"蜗牛爬"，训练慢如龟
- 学习率太大 → 模型"醉汉走路"，错过最优解
- **中期问题**：训练到一定阶段后，固定学习率会让模型"原地打转"

### PyTorch调度器（Scheduler）核心策略

#### 📈 常用官方调度器
| 类型 | 工作原理 | 适用场景 | 比喻 |
|------|----------|----------|------|
| **StepLR** | 每N轮下降固定倍数 | 基础任务 | "定期打折" |
| **ReduceLROnPlateau** | 指标停滞时自动降 | 精调模型 | "遇阻自动刹车" |
| **CosineAnnealing** | 余弦曲线式衰减 | 竞赛冲榜 | "先快后慢的刹车" |
| **OneCycleLR** | 单周期先升后降 | 快速收敛 | "先冲刺再精细调整" |

#### ⚙️ 使用要点
1. **调用顺序**：必须在`optimizer.step()`**之后**调用`scheduler.step()`
   - 错误顺序 = "先调油门再踩刹车"
2. **多调度器**：可串联使用（如先OneCycleLR再ReduceLROnPlateau）

#### 🔧 自定义调度器
当官方不满足需求时（如"每30轮降为1/10"）：
- 直接修改优化器的`lr`参数
- 核心函数：`adjust_learning_rate(optimizer, epoch)`
- 像"手动编写调速程序"，完全掌控学习节奏

> 💡 **最佳实践**：初期用StepLR快速探索，后期用ReduceLROnPlateau精细调优

---

## 三、模型微调利器：timm库

### 为什么用timm？
- **torchvision的超级扩展版**：592+预训练模型（截至2022）
- **SOTA模型仓库**：包含ResNet、ViT等最新架构
- **统一接口**：比手动实现预训练模型省时80%

### 核心操作指南
#### 🔍 查看可用模型
```python
# 查所有预训练模型
timm.list_models(pretrained=True) 

# 模糊搜索（如所有densenet）
timm.list_models("*densenet*") 
```

#### 🛠️ 常用操作
| 操作 | 代码示例 | 说明 |
|------|----------|------|
| **加载模型** | `timm.create_model('resnet34', pretrained=True)` | 自动下载预训练权重 |
| **修改输出层** | `...num_classes=10` | 适配你的分类任务 |
| **调整输入通道** | `...in_chans=1` | 单通道图像（如X光片） |
| **查看模型参数** | `model.default_cfg` | 获取输入尺寸/归一化参数 |

#### 💾 模型保存与加载
- 与PyTorch原生方式完全兼容：
  ```python
  torch.save(model.state_dict(), 'model.pth')
  model.load_state_dict(torch.load('model.pth'))
  ```

> 💡 **关键提示**：使用前必看`default_cfg`！它告诉你：
> - 输入图片尺寸（如224×224）
> - 归一化参数（mean/std）
> - 避免"喂错食物给模型"

---

## 四、半精度训练：显存"瘦身"秘籍

### 什么是半精度？
- **常规精度**（float32）：32位存储，像"高清地图"，精确但占空间
- **半精度**（float16）：16位存储，像"简略地图"，省空间但略有模糊
- **关键事实**：深度学习中多数计算不需要float32精度

### 为什么用半精度？
| 优势 | 说明 |
|------|------|
| **显存减半** | batch size可增大2倍，训练更快 |
| **计算加速** | GPU张量核心专为float16优化 |
| **避免OOM** | 解决"显存不足"的常见痛点 |

### 三步开启半精度
1. **导入工具**：`from torch.cuda.amp import autocast`
2. **模型装饰**：在`forward`函数前加`@autocast()`
3. **训练包裹**：数据进模型前加`with autocast():`

```python
for x in train_loader:
    x = x.cuda()
    with autocast():  # 自动切换精度
        output = model(x)
        loss = criterion(output, y)
```

### ⚠️ 注意事项
- **适用场景**：大数据量任务（3D图像/视频），小数据集（如MNIST）收益有限
- **精度安全**：关键计算（如softmax）会自动转回float32
- **硬件要求**：需支持Tensor Core的GPU（如V100/T4及以上）

> 💡 **经验法则**：当batch size被显存卡住时，半精度是首选解决方案

---

## 五、数据增强神器：imgaug

### 为什么需要数据增强？
- **核心问题**：真实数据太少 → 模型"死记硬背"（过拟合）
- **解决思路**：用算法生成"合理变体"，让数据集"以假乱真"
- **类比**：教孩子认猫 → 不仅给标准照片，还给旋转/模糊/裁剪版

### imgaug vs torchvision.transforms
| 特性 | imgaug | torchvision |
|------|--------|-------------|
| **增强种类** | 100+种 | 基础30+种 |
| **空间变换** | 专业级（弹性变形等） | 基础支持 |
| **医学图像** | 更友好 | 需额外处理 |
| **学习曲线** | 稍陡 | 较平缓 |

### 核心操作流程
#### 🖼️ 单图增强
1. 读取图片 → `imageio.imread()`
2. 构建增强流水线：
   ```python
   aug_seq = iaa.Sequential([
       iaa.Affine(rotate=(-25,25)),  # 随机旋转
       iaa.AdditiveGaussianNoise(),  # 加噪声
       iaa.Crop(percent=(0,0.2))     # 随机裁剪
   ])
   ```
3. 应用增强 → `aug_seq(image=img)`

#### 📦 批量增强技巧
- **统一处理**：`images_aug = aug_seq(images=[img1, img2, ...])`
- **差异化处理**：用`iaa.Sometimes(0.5, then_list=..., else_list=...)`
  - 50%图片走A流程，50%走B流程
  - 像"给不同学生布置不同作业"

#### 🔌 PyTorch集成方案
```python
transform = transforms.Compose([
    iaa.Sequential([...]).augment_image,  # imgaug处理
    transforms.ToTensor()                  # 转回Tensor
])
```
> **关键提示**：多进程训练时需设置`worker_init_fn`保证随机性

---

## 六、超参数管理：argparse实战指南

### 为什么需要argparse？
- **痛点**：硬编码超参数 → 修改需改代码 → 容易出错
- **理想状态**：`python train.py --lr 1e-4 --batch_size 64`
- **类比**：像汽车仪表盘，随时调节"油门/刹车"

### 三步掌握argparse
#### 1️⃣ 定义参数（config.py）
```python
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='学习率，太小慢，太大飘')
parser.add_argument('--batch_size', type=int, required=True,
                    help='必须指定！显存能吃几碗饭')
```

#### 2️⃣ 解析参数
```python
opt = parser.parse_args()
print(f"使用学习率: {opt.lr}")
```

#### 3️⃣ 项目集成
- **训练脚本**：`train.py`中导入`opt = config.get_options()`
- **关键参数**：
  - `required=True`：必填参数（如batch_size）
  - `action='store_true'`：开关型参数（如--use_gpu）

### 🌟 高效实践技巧
| 场景 | 解决方案 | 优势 |
|------|----------|------|
| **复现实验** | 固定`--seed 118` | 结果可重现 |
| **服务器训练** | `--checkpoint_path ./model.pth` | 断点续训 |
| **参数分组** | 按功能分config文件 | 代码整洁 |

> 💡 **终极建议**：复杂项目用YAML/JSON管理参数，argparse作入口

---

## 总结：深度学习效率工具箱

| 工具 | 核心价值 | 一句话口诀 |
|------|----------|------------|
| **自定义Loss** | 任务适配 | "标准尺不够用，自己造把尺" |
| **Scheduler** | 动态调优 | "学习步幅要变化，后期得踩刹车" |
| **timm** | 模型仓库 | "592个SOTA模型，拿来即用" |
| **半精度** | 显存优化 | "float16省一半，训练快如闪电" |
| **imgaug** | 数据增强 | "数据太少别发愁，算法造出新样本" |
| **argparse** | 参数管理 | "超参数不硬写，命令行随时调" |

> **学习建议**：先掌握每个工具的"核心30%"（本文重点），遇到具体问题再深入细节。工具是为解决问题服务，不必追求一次性掌握全部API。