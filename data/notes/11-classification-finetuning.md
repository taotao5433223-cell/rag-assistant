# 11. 文本分类微调

> 对应章节：第 6 章（6.1–6.9）

## 一、本章定位

第 4-5 章实现了 GPT 架构并预训练（或加载 OpenAI 权重）。第 6 章进入 **第三阶段第 (8) 步：将预训练 LLM 微调为文本分类器**——以垃圾消息识别为示例。

本章覆盖：

- 不同类型微调
- 准备分类数据集
- 修改预训练模型
- 微调识别垃圾消息
- 评估分类准确率
- 对新数据分类

## 二、不同类型的微调

### 2.1 两种主流方法

- **指令微调（instruction fine-tuning）**：用"指令-答案"对训练，提升模型理解并执行自然语言指令的能力。适合多种任务，需要更大数据集和更多计算资源。
- **分类微调（classification fine-tuning）**：把模型训练来识别一组特定类别标签。少数据少算力，但应用范围局限于训练时遇到的类别。

### 2.2 关键边界

经过分类微调的模型 **只能预测它在训练中遇到的类别**——它能判断"垃圾/非垃圾"，但不能对输入文本做其他分析或说明。

> **重要认知**：分类微调模型输出空间锁死在 N 维，不能当通用模型用。这是下章指令微调要解决的问题。

### 2.3 本章用分类微调的原因

- 数据少、计算资源少
- 任务明确（二分类）
- 是入门微调的最简单路径

## 三、准备数据集

### 3.1 SMSSpamCollection

使用包含垃圾消息和非垃圾消息的文本消息数据集：

```
ham    4825 条
spam    747 条
```

类别严重不平衡。

### 3.2 下采样平衡

为简单起见，下采样使每类 747 条：

```python
n_spam = len(spam_df)
balanced_df = pd.concat([spam_df, ham_df.sample(n=n_spam, random_state=42)])
# ham    747
# spam   747
```

类别标签转整数：`ham → 0, spam → 1`。

### 3.3 三分数据集

按机器学习惯例：70% 训练 / 10% 验证 / 20% 测试。保存为 CSV 便于重用。

## 四、创建数据加载器

### 4.1 填充策略

不同消息长度不同，需填充到统一长度做批处理。两种方案：

- 截断到最短消息长度 → 信息丢失
- 填充到最长消息长度 → 保留信息

本书选第二种，用 `装。

### 4.2 SpamDataset 类

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)  # 120
        else:
            self.max_length = max_length
        
        self.encoded_texts = [
            encoded[:self.max_length]  # 截断
            + [pad_token_id] * (self.max_length - len(encoded[:self.max_length]))  # 填充
            for encoded in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded), torch.tensor(label)
```

- 训练集最长序列：**120 词元**
- 验证集/测试集：填充/截断到与训练集相同的 max_length（120）

### 4.3 关键工程决策：填充到最长训练序列

**不要把输入填充到模型最大上下文（1024）**——这会降低分类性能。原因：

- 大量样本真实长度远小于 1024，填充到 1024 = 大量无关填充
- 模型要在大量噪声中找信号，准确率反降

填充到"批次内最长"或"训练集最长"是更优选择。

### 4.4 数据加载器

```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)
# 130 training batches / 19 val / 38 test
```

## 五、初始化带预训练权重的模型

### 5.1 加载 GPT-2 small (124M)

```python
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

加载后用 `generate_text_simple` 生成文本，验证权重已正确加载：

```
输入: "Every effort moves you"
输出: "Every effort moves you forward. The first step..."
```

### 5.2 预训练模型无法遵循指令

测试让模型分类垃圾消息：

```
输入: "Is the following text 'spam'? Answer with 'yes' or 'no': ..."
输出: （模型只是续写问题，不会回答）
```

这是预期内的——模型只经过预训练，缺乏指令微调。本章用分类微调解决（而非指令微调）。

## 六、添加分类头

### 6.1 替换输出层

原始输出层：`Linear(768, 50257)`（映射到词汇表）。
分类输出层：`Linear(768, 2)`（映射到 2 个类别）。

```python
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],  # 768
    out_features=num_classes
)
```

输出形状从 `[batch, seq_len, 50257]` 变为 `[batch, seq_len, 2]`。

### 6.2 只关注最后一个词元

GPT 模型的因果注意力掩码让每个位置只看自己和之前的位置。**最后一个词元是唯一能访问之前所有数据的词元**——它累积了最多信息。

所以分类微调只关注最后一个输出词元：

```python
outputs = model(inputs)
last_token_output = outputs[:, -1, :]  # (batch, 2)
```

### 6.3 冻结底层，只微调靠近输出的层

模型已预训练，不需要全量微调。**冻结所有层，只解冻最后Transformer 块 + final_norm + 输出头**：

```python
# 先全部冻结
for param in model.parameters():
    param.requires_grad = False

# 替换输出层（默认 requires_grad=True）
model.out_head = nn.Linear(768, 2)

# 解冻最后一个 Transformer 块和最终 LayerNorm
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
```

理由：较低层捕捉基本语言结构和语义，适用于广泛任务；最后几层更侧重特定任务特征。只微调最后几层既够用又计算高效。

## 七、计算分类损失与准确率

### 7.1 损失

与预训练类似用交叉熵，但 targets 是类别标签（0/1）而非下一词元 ID：

```python
loss = torch.nn.functional.cross_entropy(
    last_token_output,  # (batch, 2)
    target_labels       # (batch,)
)
```

### 7.2 准确率

```python
predicted_labels = torch.argmax(last_token_output, dim=-1)  # softmax + argmax
accuracy = (predicted_labels == target_labels).float().mean()
```

### 7.3 关键边界：准确率不可微

分类准确率是离散指标，**不能直接当训练目标**——优化器只能优化可微的损失函数。所以训练用交叉熵损失，评估用准确率。

### 7.4 训练前的预期

未微调模型的初始准确率约 50%（二分类随机猜）。微调后预期 95%+。

## 八、微调训练

### 8.1 训练配置

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
num_epochs = 5  # 微调 5 轮是不错起点
```

训练循环与第 5 章类似，但目标和分类标签而非下一词元。

### 8.2 结果

```
Epoch 5: train_loss 0.05, val_loss 0.12
         train_acc 97.21%, val_acc 95.67%
```

5 轮达到 97.21% 训练准确率 / 95.67% 测试准确率。

## 九、使用微调后的模型

### 9.1 对新数据分类

```python
def classify_message(model, tokenizer, msg, max_length, pad_token_id=50256):
    model.eval()
    encoded = tokenizer.encode(msg)
    encoded = encoded[:max_length] + [pad_token_id] * (max_length - len(encoded[:max_length]))
    input_tensor = torch.tensor(encoded).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "Spam" if predicted_label == 1 else "Not Spam"
```

### 9.2 警示：分类微调模型不能做别的

```python
# 错误用法
model("解释这段代码")  # 分类微调模型只会输出 0 或 1，不会解释
```

分类微调模型 **输出空间锁死在 2 维**，不能当通用模型用。要聊天/解释代码，需要指令微调（下章）。

## 十、第 6 章关键决策回顾

| 决策点 | 选择 | 理由 |
|---|---|---|
| 微调类型 | 分类微调 | 数据少、任务明确、入门最简 |
| 填充策略 | 填到训练集最长（120） | 填到 1024 反降性能 |
| 输出层 | 替换为 768→2 | 任务决定输出维度 |
| 关注词元 | 最后一个 | 因果掩码让它累积最多信息 |
| 微调层 | 输出头 + 最后TrfBlock + final_norm | 底层已通用，只调顶层 |
| 损失函数 | 交叉熵 | 准确率不可微，不能直接当目标 |
| 训练轮数 | 5 轮 | 微调 5 轮是不错起点 |

## 十一、本篇要点

- 分类微调只输出训练时见过的类别——**输出空间锁死 N 维，不能当通用模型用**。
- 数据集不平衡需下采样平衡；按 70/10/20 划分训练/验证/测试。
- 填充策略：填到训练集最长（120），不要填到模型最大上下文（1024）——填充越多反而越差。
- 加载预训练 GPT-2 权重，替换输出层 768→2。
- 只关注最后一个词元——因果掩码让它累积最多信息。
- 冻结底层，只解冻最后Transformer 块 + final_norm + 输出头。
- 准确率不可微，训练用交叉熵、评估用准确率。
- 5 轮训练达到 97% 训练准确率 / 96% 测试准确率。
- 微调后模型只能分类，不能聊天/解释代码——那是下章指令微调的事。
