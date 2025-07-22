---
文档名-title: HappyLLm课程笔记章节03.1 - 注意力机制
创建时间-create time: 2025-07-22 20:15
更新时间-modefived time: 2025-07-22 20:15 星期二
文档粗分-text: 笔记
笔记细分-text: 
笔记索引-link: '[[笔记总索引]]'
继承自-link: 
tags:
  - 笔记
模板自: -笔记-规范（2024.6.8）
---

##  对Transformer的优化思路
针对 Encoder、Decoder 的特点，引入 ELMo 的预训练思路，开始出现不同的、对 Transformer 进行优化的思路。

Google 仅选择了 Encoder 层，通过将 Encoder 层进行堆叠，再提出不同的预训练任务-掩码语言模型（Masked Language Model，MLM），打造了一统自然语言理解（Natural Language Understanding，NLU）任务的代表模型——BERT。而 OpenAI 则选择了 Decoder 层，使用原有的语言模型（Language Model，LM）任务，通过不断增加模型参数和预训练语料，打造了在 NLG（Natural Language Generation，自然语言生成）任务上优势明显的 GPT 系列模型，也是现今大火的 LLM 的基座模型。


当然，还有一种思路是同时保留 Encoder 与 Decoder，打造预训练的 Transformer 模型，例如由 Google 发布的 T5模型。

## BERT

全名为 Bidirectional Encoder Representations from Transformers

。自 BERT 推出以来，预训练+微调的模式开始成为自然语言处理任务的主流，不仅 BERT 自身在不断更新迭代提升模型性能，也出现了如 MacBERT、BART 等基于 BERT 进行优化提升的模型。

### 主导

可以说，BERT 是自然语言处理的一个阶段性成果，标志着各种自然语言处理任务的重大进展以及预训练模型的统治地位建立，一直到 LLM 的诞生，NLP 领域的主导地位才从 BERT 系模型进行迁移。即使在 LLM 时代，要深入理解 LLM 与 NLP，BERT 也是无法绕过的一环。

###  推动训练和微调范式

而 BERT 也采用了该范式，并通过将模型架构调整为 Transformer，引入更适合文本理解、能捕捉深层双向语义关系的预训练任务 MLM，将预训练-微调范式推向了高潮。

![[Pasted image 20250722202207.png]]

### BERT - 输入自然语言，输出矩阵（而不是自然语言） 的Encoder

BERT 是针对于 NLU （自然语言理解）任务打造的预训练模型，其输入一般是文本序列，而输出一般是 Label，例如情感分类的积极、消极 Label。但是，正如 Transformer 是一个 Seq2Seq 模型，使用 Encoder 堆叠而成的 BERT 本质上也是一个 Seq2Seq 模型，只是没有加入对特定任务的 Decoder，因此，为适配各种 NLU 任务，在模型的最顶层加入了一个分类头 prediction_heads，用于将多维度的隐藏状态通过线性层转换到分类维度（例如，如果一共有两个类别，prediction_heads 输出的就是两维向量）。

输入的文本序列会首先通过 tokenizer（分词器） 转化成 input_ids（基本每一个模型在 tokenizer 的操作都类似，可以参考 Transformer 的 tokenizer 机制，后文不再赘述），然后进入 Embedding 层转化为特定维度的 hidden_states，再经过 Encoder 块。Encoder 块中是对叠起来的 N 层 Encoder Layer，BERT 有两种规模的模型，分别是 base 版本（12层 Encoder Layer，768 的隐藏层维度，总参数量 110M），large 版本（24层 Encoder Layer，1024 的隐藏层维度，总参数量 340M）。通过Encoder 编码之后的最顶层 hidden_states 最后经过 prediction_heads 就得到了最后的类别概率，经过 Softmax 计算就可以计算出模型预测的类别。

### 输入的文本序列转化成类别

![[Pasted image 20250722202539.png]]

输入的文本序列会首先通过 tokenizer（分词器） 转化成 input_ids（基本每一个模型在 tokenizer 的操作都类似，可以参考 Transformer 的 tokenizer 机制，后文不再赘述），然后进入 Embedding 层转化为特定维度的 hidden_states，再经过 Encoder 块。Encoder 块中是对叠起来的 N 层 Encoder Layer，BERT 有两种规模的模型，分别是 base 版本（12层 Encoder Layer，768 的隐藏层维度，总参数量 110M），large 版本（24层 Encoder Layer，1024 的隐藏层维度，总参数量 340M）。通过Encoder 编码之后的最顶层 hidden_states 最后经过 prediction_heads 就得到了最后的类别概率，经过 Softmax 计算就可以计算出模型预测的类别。

### prediction_heads预测头，起到转换成概率的作用

![[Pasted image 20250722202753.png]]

prediction_heads 其实就是线性层加上激活函数，一般而言，最后一个线性层的输出维度和任务的类别数相等，如图3.3所示：

###  Encoder Layer和T中的类似

而每一层 Encoder Layer 都是和 Transformer 中的 Encoder Layer 结构类似的层，如图3.4所示：

https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-2.png

### BERT - 特殊层残差和源输入

![[Pasted image 20250722203030.png]]

如图3.5所示，已经通过 Embedding 层映射的 hidden_states 进入核心的 attention 机制，然后通过残差连接的机制和原输入相加，再经过一层 Intermediate 层得到最终输出。Intermediate 层是 BERT 的特殊称呼，其实就是一个线性层加上激活函数：

### GELU函数

注意，BERT 所使用的激活函数是 GELU 函数，全名为高斯误差线性单元激活函数，这也是自 BERT 才开始被普遍关注的激活函数。GELU 的计算方式为：

GELU(x)=0.5x(1+tanh(2π)(x+0.044715x3))

GELU 的核心思路为将随机正则的思想引入激活函数，通过输入自身的概率分布，来决定抛弃还是保留自身的神经元。关于 GELU 的原理与核心思路，此处不再赘述，有兴趣的读者可以自行学习。

![[Pasted image 20250722203121.png]]


BERT 的 注意力机制和 Transformer 中 Encoder 的 自注意力机制几乎完全一致，但是 BERT 将相对位置编码融合在了注意力机制中，将相对位置编码同样视为可训练的权重参数，如图3.6所示：

如图，BERT 的注意力计算过程和 Transformer 的唯一差异在于，在完成注意力分数的计算之后，先通过 Position Embedding 层来融入相对位置信息。这里的 Position Embedding 层，其实就是一层线性矩阵。通过可训练的参数来拟合相对位置，相对而言比 Transformer 使用的绝对位置编码 Sinusoidal 能够拟合更丰富的相对位置信息，但是，这样也增加了不少模型参数，同时完全无法处理超过模型训练长度的输入（例如，对 BERT 而言能处理的最大上下文长度是 512 个 token）。

### 总结，模型架构确实是建立在Transformer 的 Encoder 之上