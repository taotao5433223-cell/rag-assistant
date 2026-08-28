# 04. BPE 与滑动窗口采样

> 对应章节：第 2 章 2.4–2.5（含数据加载器、嵌入层实现）

## 一、引入特殊上下文词元

### 1.1 解决未知词问题

为处理未知词，向词汇表添加两个特殊词元：

- `<|unk|>`：表示未出现在训练数据中的新词/未知词。
- `<|endoftext|>`：分隔两个不相关的文本来源（如多篇文档拼接时标识边界）。

```python
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# 词汇表大小：1130 → 1132
```

### 1.2 SimpleTokenizerV2

新分词器把未知词替换为 `<|unk|>`：

```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# 'Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.'

ids = tokenizer.encode(text)
# 包含 1130（<|endoftext|>）和 1131（<|unk|>）

print(tokenizer.decode(ids))
# '<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.'
```

"Hello" 和 "palace" 在《The Verdict》中未出现，被替换为 `<|unk|>`。

### 1.3 常见特殊词元

不同 LLM 可能引入的特殊词元：

- **[BOS]**（Beginning Of Sequence）：标记文本起点。
- **[EOS]**（End Of Sequence）：位于文本末尾，类似 `<|endoftext|>`，特别适用于连接多个不相关文本。
- **[PAD]**（Padding）：批次大小>1 时文本长度不一，较短文本用 [PAD] 填充以匹配最长文本。

**GPT 模型的选择**：

- 只使用 `<|endoftext|>`，简化流程；它兼具 [EOS] 作用，也用于填充。
- **不使用 `<|unk|>`**——而是用 **BPE 分词器**把单词拆解为子词单元，从根上避免未知词。

## 二、BPE 字节对编码

### 2.1 为什么需要 BPE

单词级分词器有致命缺陷：训练集没有的词就报 KeyError。BPE 通过把未知词拆成已知的子词/字符，让分词器能处理任何输入而不崩。

BPE 用于训练 GPT-2、GPT-3、ChatGPT 原始模型。

### 2.2 用 tiktoken 实现 BPE

BPE 实现相对复杂，本书使用开源库 `tiktoken`（基于 Rust，高效实现 BPE）：

```bash
pip install tiktoken
```

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```

用法与 SimpleTokenizerV2 相似：

```python
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 466, ...]
```

### 2.3 BPE 处理未知词的机制

BPE 把未在词汇表的词拆成子词单元。比如 "sunlit" 可能被拆成 "sun" + "lit"；完全没见过的词最终能拆到字符级别。这样 **任何文本都能被编码**，不再有 KeyError。

GPT-2 的 BPE 词汇表大小为 **50257**（远大于《The Verdict》的 1132）。

### 2.4 BPE vs 单词级分词器

| 对比项 | 单词级分词器 | BPE |
|---|---|---|
| 未知词处理 | KeyError | 拆成子词，不崩 |
| 词汇表大小 | 1130（受训练集限制） | 50257（GPT-2 通用） |
| 适配场景 | 教学/小型实验 | 生产级 LLM |
| GPT 是否使用 | 否 | 是（GPT-2/3/ChatGPT） |

## 三、数据采样与滑动窗口

### 3.1 为什么需要滑动窗口

LLM 训练任务是"下一词预测"：给定一段文本，预测下一个词。我们需要从长文本中切出"输入-目标"对：

- 输入：一段文本（如 4 个词）
- 目标：相同文本向后移一位（如 4 个词，第一个是被预测的下一词）

这就是 **滑动窗口（sliding window）** 采样。

### 3.2 滑动窗口的实现思路

`create_dataloader_v1` 函数把长文本切成定长块，每块生成 (input, target) 对：

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

### 3.3 关键参数：max_length 与 stride

- **max_length**：每个输入块的长度（如 4 词元的示例或 256 词元的实际训练）。
- **stride**：窗口每次移动的步长。

**stride 的关键选择**：

- `stride = max_length`（推荐）：窗口不重叠，每个词元只在训练集中出现一次（作为某 input 的一部分），既不跳词又避免批次重叠增加过拟合风险。
- `stride < max_length`（重叠采样）：充分利用数据，但同一词元会在多个训练样本里出现，**会增加过拟合风险**——小数据集尤为明显。
- `stride > max_length`：会跳过部分词元，造成数据浪费。

本书预训练示例用 `max_length = GPT_CONFIG_124M["context_length"]`（实际 256，为笔记本可跑），`stride = max_length`。

### 3.4 DataLoader 批处理

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)
    return dataloader
```

`batch_size` 把多个 (input, target) 对组成批次，提高训练效率。`shuffle=True` 打乱顺序，`drop_last=True` 丢弃最后不完整的批次。

## 四、词元嵌入层实现

### 4.1 词元 ID → 向量

得到词元 ID 后，需要把它们转换为嵌入向量。PyTorch 的 `nn.Embedding` 是查表操作：给定词元 ID，返回对应的嵌入向量。

```python
import torch.nn as nn
vocab_size = 50257
output_dim = 256  # 嵌入维度（教学用，GPT-2 small 实际 768）
embedding_layer = nn.Embedding(vocab_size, output_dim)
```

`nn.Embedding` 的权重矩阵形状是 `(vocab_size, output_dim)`，即 50257 × 256。这些权重是 **可训练的**——在 LLM 训练中会与模型其他参数一起优化。

### 4.2 位置嵌入（Positional Embedding）

LLM 的自注意力机制本身对位置无感（它看到的是一组词元，不知道谁先谁后）。所以需要 **位置嵌入** 给每个位置加上位置信息。

- **绝对位置嵌入**：为每个位置（0 到 context_length-1）学一个固定向量，加到词元嵌入上。GPT 选这种。
- **相对位置嵌入**：编码词元间的相对距离（本书不实现）。

GPT-2 small 配置 `context_length = 1024`，所以位置嵌入层是 `nn.Embedding(1024, output_dim)`。

```python
context_length = 1024
pos_embedding_layer = nn.Embedding(context_length, output_dim)
```

最终输入 = 词元嵌入 + 位置嵌入：

```python
token_embeddings = embedding_layer(input_ids)      # (batch, seq_len, emb_dim)
pos_embeddings = pos_embedding_layer(torch.arange(seq_len))  # (seq_len, emb_dim)
input_embeddings = token_embeddings + pos_embeddings
```

## 五、第 2 章小结

第 2 章完整走完了 **第一阶段第 (1) 步：数据采样流水线**：

```
原始文本 (the-verdict.txt)
    ↓ 分词（正则 re.split）
词元列表
    ↓ BPE（tiktoken gpt2）
词元 ID 序列
    ↓ 滑动窗口（max_length, stride=max_length）
(input, target) 对
    ↓ DataLoader 批处理
训练批次 (batch_size, max_length)
    ↓ 词元嵌入 + 位置嵌入
输入嵌入张量 (batch_size, seq_len, emb_dim)
    ↓ 进入第 3 章：注意力机制
```

## 六、关键概念速查

| 术语 | 含义 |
|---|---|
| `<\|unk\|>` | 未知词占位符（GPT 不用，改用 BPE） |
| `<\|endoftext\|>` | 文本分隔/结束符，词元 ID 50256（GPT 也用它做填充） |
| BPE | 字节对编码，把未知词拆成子词，GPT-2/3/ChatGPT 使用 |
| tiktoken | OpenAI 的 BPE 分词器库，GPT-2 词汇表 50257 |
| 滑动窗口 | 从长文本切出 (input, target) 对，input 是文本片段，target 是 input 向后移一位 |
| stride | 窗口移动步长；推荐 stride = max_length 避免重叠过拟合 |
| 词元嵌入 | `nn.Embedding(vocab_size, emb_dim)`，把词元 ID 查表为向量，可训练 |
| 位置嵌入 | 给每个位置加位置信息；GPT 用绝对位置嵌入，`nn.Embedding(context_length, emb_dim)` |

## 七、本篇要点

- GPT 不用 `<|unk|>`，改用 BPE 拆子词，任何输入都能编码不崩。
- BPE 词汇表 50257（GPT-2），通过 tiktoken 库使用。
- 滑动窗口切 (input, target) 对：target 是 input 向后移一位。
- **stride = max_length** 是关键选择——不跳词又避免批次重叠过拟合（小数据集尤其重要）。
- 词元嵌入 + 位置嵌入 = LLM 输入；两者都是可训练的 `nn.Embedding`。
- GPT 用绝对位置嵌入，`context_length=1024`。
- 第 2 章完成"原始文本→训练批次→输入嵌入"的完整数据流水线。
