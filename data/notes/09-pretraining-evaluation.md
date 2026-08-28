# 09. 预训练评估：交叉熵与困惑度

> 对应章节：第 5 章 5.1

## 一、本章定位

第 4 章实现了 GPT 模型架构，本章（第 5 章）实现训练函数并对 LLM 进行预训练。本章关注 **第二阶段：预训练 LLM**，覆盖：

- 计算训练集和验证集损失，评估生成文本质量
- 实现训练函数并预训练 LLM
- 保存和加载模型权重
- 从 OpenAI 加载预训练权重

本篇（5.1）聚焦 **评估生成文本质量**——这是训练过程中优化 LLM 的必要条件。没有数值评估，就无法监测训练是否在改善。

## 二、文本生成回顾

### 2.1 generate_text_simple 函数

第 4 章实现的 `GPTModel` 接收词元 ID，输出 logits（形状 `[batch, num_tokens, vocab_size]`）。`generate_text_simple` 用它逐词生成：

```
1. 分词器把输入文本 → 词元 ID
2. 模型接收词元 ID → 输出 logits（词汇表每词元的概率分数）
3. logits → softmax → 概率 → argmax → 下一个词元 ID
4. 把新词元 ID 拼到输入，重复
5. 词元 ID → 文本
```

### 2.2 未训练模型的输出

```python
model = DummyGPTModel(GPT_CONFIG_124M)
# 输入 "Every effort moves you"
# 输出 "Every effort moves you rentingetic wasn? refres RexMeCHicular stren"
```

模型还没训练，所以输出乱码。要判断"什么是连贯/高质量的文本"，必须用 **数值方法** 评估——这样训练过程才能监测和优化。

## 三、计算文本生成损失

### 3.1 输入与目标

回顾第 2 章：targets 是 inputs 向后移一位。这种移位指导模型预测序列中的下一个词元。

```python
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                      [40, 1107, 588]])       # "I really like"]
targets = torch.tensor([[3626, 6100, 345],    # ["effort moves you",
                       [1107, 588, 11311]])   # "really like chocolate"]
```

### 3.2 损失计算的六步

模型先输出 logits（形状 `[2, 3, 50257]`），softmax 转概率。我们关心的是 **与目标词元对应的概率**：

```
步骤 ❶ 模型前向，得 logits
步骤 ❷ softmax(logits) → 概率 probas
步骤 ❸ 取出目标词元对应的概率 target_probas
步骤 ❹ 对概率取对数 log_probas（数学上更易优化）
步骤 ❺ 求平均 avg_log_probas
步骤 ❻ 取负得 neg_avg_log_probas（深度学习的习惯是把损失降为 0）
```

### 3.3 交叉熵损失

步骤 ❻ 得到的"负平均对数概率"在深度学习中称为 **交叉熵损失（cross-entropy loss）**——衡量模型预测分布与真实分布的差异。

PyTorch 内置 `cross_entropy` 函数自动完成 ❸-❻：

```python
# 展平批次维度
logits_flat = logits.flatten(0, 1)   # (6, 50257)
targets_flat = targets.flatten()      # (6,)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# tensor(10.7940)
```

未训练模型的损失约 10.79，对应每步在 48725 个词元间随机猜——离正确答案很远。

### 3.4 交叉熵的本质

- 衡量两个概率分布的差异：真实分布（目标词元）vs 模型预测分布（logits→softmax）。
- PyTorch 中"交叉熵"与"负平均对数概率"常互换使用。
- 训练目标：让正确词元的概率最大化 = 让交叉熵损失最小化。

## 四、困惑度（Perplexity）

### 4.1 公式

```
perplexity = torch.exp(loss)
```

未训练模型：`exp(10.7940) = 48725.82`。

### 4.2 直觉

困惑度表示"模型在每一步对有效词汇量的不确定性"。48725 意味着模型不确定在词汇表的 48725 个词元中应该选哪个——离确定答案很远。

- 困惑度低 → 模型预测更接近实际分布。
- 困惑度比原始损失值更易解释。

## 五、训练集与验证集损失

### 5.1 数据准备

用《The Verdict》（第 2 章用过的文本，20479 字符，5145 词元）作为训练数据。90/10 拆分：

```python
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
```

用第 2 章的 `create_dataloader_v1` 创建训练/验证加载器，`max_length=256`（实际配置），`stride=256`（避免重叠过拟合）。

### 5.2 计算加载器损失

`calc_loss_loader` 遍历加载器所有批次，累积损失并求平均：

```python
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
# Training loss: 10.9876
# Validation loss: 10.9811
```

未训练模型两个损失都约 11，对应困惑度约 60000——模型完全不知道下一个词是什么。

### 5.3 为什么需要训练集和验证集两个损失

- **训练损失**：模型在训练数据上的表现。
- **验证损失**：模型在未见过的数据上的表现——这才是泛化能力的指标。

只看训练损失会被"积极信号偏误"误导：训练 loss 下降不代表模型变好，可能只是死记硬背。**对照两条曲线** 是判断过拟合的关键（详见下篇）。

## 六、预训练成本的现实

为了使项目规模更具体，训练 Llama 2 70B 模型：

- 处理 2 万亿词元
- 在 A100 GPU 上训练 184,320 GPU 小时
- AWS 8×A100 服务器约 30 美元/小时
- 总训练成本约 **69 万美元**

本书用小数据集（5145 词元）能在几分钟内跑完，是教学简化。第 5.5 节会学习加载 OpenAI 预训练权重以跳过昂贵预训练。

## 七、关键概念速查

| 术语 | 含义 |
|---|---|
| logits | 未经 softmax 的模型输出，每维对应词汇表一个词元 |
| softmax | 把 logits 转概率分布（总和 1，全正） |
| target_probas | 与目标词元对应的概率，训练目标是最大化它 |
| 交叉熵损失 | 负平均对数概率，衡量预测分布与真实分布差异 |
| PyTorch cross_entropy | 自动完成"取目标概率→对数→平均→取负"的内置函数 |
| 困惑度 | `exp(loss)`，模型每步对有效词汇量的不确定性 |
| 训练损失 | 在训练数据上的损失 |
| 验证损失 | 在未见过数据上的损失，泛化能力的指标 |

## 八、本篇要点

- 评估生成文本必须用 **数值方法**（交叉熵/困惑度），不能凭主观判断——否则训练无从监测。
- 交叉熵 = 负平均对数概率 = 衡量预测分布与真实分布的差异。
- PyTorch `cross_entropy` 自动完成取目标概率→对数→平均→取负。
- 困惑度 = `exp(loss)`，比原始损失更易解释（48725 = 模型在 48725 个词元间随机猜）。
- 必须同时算训练损失和验证损失——只看训练 loss 会被"积极信号偏误"误导。
- 未训练模型损失约 11，困惑度约 60000。
- 预训练成本极高（Llama 2 70B 约 69 万美元）——本书用小数据集做教学，下篇学加载公开权重跳过预训练。
