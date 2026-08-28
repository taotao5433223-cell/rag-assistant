# 03. 词嵌入与文本分词

> 对应章节：第 2 章 2.1–2.3

## 一、本章在流水线中的位置

第 1 章建立了 LLM 的全景与三阶段流水线。第 2 章聚焦 **第一阶段中的第 (1) 步：数据采样流水线**——为大语言模型训练准备输入文本。

LLM 在预训练阶段一次处理一个单词，通过下一词预测任务训练拥有数百万甚至数十亿参数的模型。但在实现和训练 LLM 之前，必须先准备好训练数据集。

本章你将学习：

- 将文本分割为独立的单词词元和子词词元
- 使用更高级的 BPE 分词方法（下篇详述）
- 利用滑动窗口方法对训练样本采样（下篇详述）
- 将词元转换为输入到 LLM 的向量表示

## 二、理解词嵌入

### 2.1 为什么需要嵌入

包括 LLM 在内的深度神经网络模型 **无法直接处理原始文本**。文本数据是离散的，无法直接用它执行神经网络训练所需的数学运算。我们需要一种将单词表示为连续值向量格式的方法——这就是 **嵌入（embedding）**。

嵌入的本质：将离散对象（如单词、图像、文档）映射到连续向量空间中的点。主要目的是把非数值数据转换为神经网络可处理的格式。

### 2.2 不同数据格式需要不同嵌入模型

视频、音频、文本等原始格式都需要通过嵌入模型转换为密集向量表示。但 **不同的数据格式需要不同的嵌入模型**——为文本设计的嵌入模型不适用于音频或视频。

### 2.3 词嵌入的常见形式

词嵌入是文本嵌入中最常见的形式。也存在针对句子、段落乃至整个文档的嵌入技术（在 RAG 检索增强生成领域很流行），但本书目标是训练类 GPT 的 LLM，专注于逐词生成文本，因此集中探讨 **词嵌入**。

### 2.4 word2vec 与 LLM 自学生嵌入

- **word2vec** 是早期最流行的词嵌入方法之一。核心思想：出现在相似上下文中的词往往具有相似含义。通过训练神经网络，根据目标词预测上下文或根据上下文预测目标词，生成词嵌入。相似概念在嵌入空间中会聚集。
- **LLM 通常自行生成嵌入**——这些嵌入是输入层的一部分，会在训练过程中更新。优势：嵌入可针对特定任务和数据优化（而 word2vec 是预训练的、固定的）。

### 2.5 嵌入维度

词嵌入的维度可从 1 维到数千维不等。更高维度有助于捕捉更细微关系，但牺牲计算效率。

- 最小 GPT-2（1.17 亿参数）嵌入维度 768
- 最大 GPT-3（1750 亿参数）嵌入维度 12288

高维嵌入难以可视化（人类感官局限于 3 维以下），但处理 LLM 时通常使用高维。

## 三、文本分词

### 3.1 分词的目标

将输入文本分割为独立的词元，这是为 LLM 生成嵌入向量所必需的预处理步骤。词元既可以是单个单词，也可以是包括标点符号在内的特殊字符。

### 3.2 训练语料：The Verdict

本书使用 Edith Wharton 的短篇小说《The Verdict》作为分词教学文本，因为它已公开发表，可自由使用。文本约 20,479 个字符，可在 GitHub 仓库 `rasbt/LLMs-from-scratch` 找到 `the-verdict.txt`。

> 注意：构建 LLM 实际需要处理数百万篇文章和成千上万本图书（数千 GB）。但出于教学目的，小规模文本足以说明核心思想，可在消费级硬件上完成。

### 3.3 用正则表达式实现简易分词器

第一步尝试：按空白字符分割文本。

```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
# ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```

问题：单词仍与标点符号相连。我们希望标点符号作为单独的列表项。同时 **不转换为小写**——大写形式有助于 LLM 区分专有名词和普通名词、理解句子结构、学会正确生成大写字母。

改进：在空白字符、逗号和句号处分割：

```python
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

进一步扩展以处理问号、引号、双破折号等特殊字符：

```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

应用到《The Verdict》全文，得到 4690 个词元。

### 3.4 关于空白字符的处理抉择

- **移除空白字符**：减轻内存和计算负担。
- **保留空白字符**：当模型需对文本精确结构敏感时（如 Python 代码对缩进和空格高敏感）。

为简化输出，本章暂时移除空白字符。后续会改用保留空白字符的方案（如 BPE）。

## 四、将词元转换为词元 ID

### 4.1 构建词汇表

将词元从 Python 字符串转换为整数表示（词元 ID）是将其转为嵌入向量前的必经步骤。需要构建一张 **词汇表（vocabulary）**——定义每个唯一词元到唯一整数值的映射。

构建步骤：

1. 将训练集全部文本分词成独立词元
2. 按字母顺序排列，删除重复
3. 聚合到词汇表

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)  # 1130
vocab = {token: integer for integer, token in enumerate(all_words)}
```

《The Verdict》的词汇表大小为 **1130**。

### 4.2 SimpleTokenizerV1 类

实现一个完整的分词器类，包含 `encode` 方法（文本→词元 ID）和 `decode` 方法（词元 ID→文本）。

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 修复标点前空格
        return text
```

测试 encode 与 decode：

```python
text = """"It's the last he painted, you know," Mrs. Gisburn said."""
ids = tokenizer.encode(text)
# [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 11, 754, 793, 7]
print(tokenizer.decode(ids))
# '" It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
```

### 4.3 KeyError：未知词问题

把分词器应用到训练集外的新样本：

```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
# KeyError: 'Hello'
```

"Hello" 在《The Verdict》中未出现，所以不在词汇表里。这凸显了使用更大且更多样化训练集扩展词汇表的必要性——也是下一节引入 **特殊上下文词元** 和 **BPE** 的动机。

## 五、关键概念速查

| 术语 | 含义 |
|---|---|
| 嵌入（embedding） | 将离散对象映射到连续向量空间，把非数值数据转为神经网络可处理格式 |
| 词嵌入 | 把单词映射为向量；word2vec 是早期方法，LLM 自学生在训练中优化嵌入 |
| 词元（token） | 文本分割的单元，可以是单词或子词/特殊字符 |
| 词元 ID | 词元在词汇表中对应的整数 |
| 词汇表（vocabulary） | 唯一词元到唯一整数的映射 |
| encode / decode | 分词器的两个方法：文本→词元 ID；词元 ID→文本 |
| KeyError / OOV | 未知词问题，单词级分词器遇到词汇表外词会报错 |

## 六、本篇要点

- 深度神经网络无法直接处理原始文本，必须先嵌入为向量。
- LLM 的嵌入是输入层的一部分，会在训练中自行更新、针对任务优化。
- 嵌入维度权衡：GPT-2 small 768 维，GPT-3 最大 12288 维。
- 简易分词器用正则 `re.split` 在标点和空白处分割，不转小写以保留专有名词信息。
- 《The Verdict》词汇表 1130 个，分词器需 encode/decode 双向能力。
- 单词级分词器遇到未知词会 KeyError——这是引入特殊词元和 BPE 的动机（下篇详述）。
