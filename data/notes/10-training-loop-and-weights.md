# 10. 训练循环、解码策略与权重管理

> 对应章节：第 5 章 5.2–5.5

## 一、训练大语言模型

### 1.1 训练循环的 8 步

典型的 PyTorch 神经网络训练流程：

```
1. 遍历每个训练轮次（epoch）
2. 遍历每个训练批次
3. 重置梯度（optimizer.zero_grad()）
4. 前向计算损失
5. 反向传播计算新梯度（loss.backward()）
6. 用梯度更新权重（optimizer.step()）
7. 监控步骤：打印损失、生成样本
8. （周期性）评估模型、保存检查点
```

### 1.2 train_model_simple 函数

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
        
        # 定期评估
        train_loss = evaluate_model(model, train_loader, device, eval_iter)
        val_loss = evaluate_model(model, val_loader, device, eval_iter)
        # 打印并生成样本
        generate_and_print_sample(model, device, start_context, tokenizer)
```

### 1.3 优化器：AdamW

本书用 **AdamW 优化器**——Adam 的变体，改进了权重衰减方法（对较大权重进行惩罚以防止过拟合）。AdamW 在 LLM 训练中常用。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
```

### 1.4 训练 10 轮的结果

在《The Verdict》上训练 10 轮（MacBook Air 约 5 分钟）：

```
Epoch 1: train_loss 9.781 → 0.391（训练集）
         val_loss   9.933 → 6.452（验证集）

训练前生成: "Every effort moves you,,,,,,,,,,,,"
训练后生成: "Every effort moves you forward. The first step..."
```

## 二、过拟合的诊断

### 2.1 训练损失 vs 验证损失发散

第 1 轮两者一起下降，但第 2 轮后开始 **发散**：

- 训练损失继续降到 0.391
- 验证损失停在 6.452

**训练损失远低于验证损失 + 验证损失停滞 → 过拟合信号**。

### 2.2 逐字记忆训练数据

通过搜索生成文本可确认模型逐字记住了训练集段落（如 "quite insensible to the irony"）。

### 2.3 为什么会过拟合

- 数据集极小（5145 词元）
- 多轮训练
- 模型容量远超数据复杂度

通常在更大数集上只训练 1 轮就很常见，能避免这种过拟合。

### 2.4 诊断准则

| 训练损失 | 验证损失 | 判断 |
|---|---|---|
| 下降 | 同步下降 | 健康训练 |
| 下降 | 停滞 | 过拟合开始 |
| 远低于验证 | 停滞/反弹 | 严重过拟合 |
| 不降 | 不降 | 学习率/架构问题 |

## 三、解码策略

训练完的模型会逐字记忆训练段落——为减少记忆、增加独创性，需要更好的 **解码策略**。

### 3.1 贪婪解码（greedy decoding）

`generate_text_simple` 用的就是贪婪解码——每步选概率最高的词元。问题：容易逐字复现训练段落，缺乏独创性。

### 3.2 温度缩放（temperature scaling）

用温度参数 τ 调节 softmax 分布的"尖锐度"：

```
scaled_logits = logits / τ
probas = torch.softmax(scaled_logits, dim=-1)
```

- **τ < 1**（如 0.1）：分布更尖锐，更确定（倾向选最高概率词元）
- **τ = 1**：原始 softmax
- **τ > 1**（如 2.0）：分布更平坦，更随机（增加多样性，但也可能产生语法错误）

```python
top_probas = torch.topk(probas, 5)  # 取概率最高的 5 个
```

### 3.3 Top-k 采样

只从概率最高的 k 个词元中采样，避免选到无意义低概率词元：

```python
top_k = 50
top_logits = torch.topk(logits, top_k)
indices_to_remove = logits < torch.min(top_logits)
logits = logits.masked_fill(indices_to_remove, -torch.inf)
probas = torch.softmax(logits, dim=-1)
```

温度 + Top-k 组合：在多样性与可读性之间找平衡——既不逐字复现，也不无意义乱语。

### 3.4 完整的 generate 函数

```python
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 只取最后一个位置
        
        if top_k is not None:
            top_logits = torch.topk(logits, top_k)
            min_val = torch.min(top_logits)
            logits = torch.masked_fill(logits, logits < min_val, -float('inf'))
        
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 采样
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 贪婪
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

`temperature=0` 退化为贪婪解码。

## 四、保存与加载模型权重

### 4.1 为什么要保存/加载

- 训练中断后从断点续跑
- 部署训练好的模型
- 分享模型

### 4.2 PyTorch 的 state_dict

PyTorch 把模型参数存储在 `state_dict` 字典里。保存：

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
```

### 4.3 加载

```python
checkpoint = torch.load("model_and_optimizer.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

**关键细节**：**同时保存 model 和 optimizer 的 state_dict**——optimizer 内部有动量等状态，不保存会让训练失去连贯性，模型可能失去生成连贯文本的能力。

## 五、从 OpenAI 加载预训练权重

### 5.1 动机

预训练太贵（Llama 2 70B 约 69 万美元）。OpenAI 公开了 GPT-2 权重，可直接加载到我们的 GPTModel 实现中——跳过昂贵预训练，直接做下游微调。

### 5.2 下载与加载

```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
```

`download_and_load_gpt2` 下载 OpenAI 的 TF checkpoint 并转成 Python 字典。

### 5.3 配置对齐（关键工程）

加载权重时必须做 **配置对齐**，一处错配静默输出乱码：

- `context_length`：OpenAI 用 1024，本书教学时砍到 256——加载前要改回 1024
- `qkv_bias`：OpenAI 的 GPT-2 用 True，本书默认 False——加载时改 True
- 命名映射：OpenAI 的 TF 变量命名与本书 PyTorch 命名不同，要做映射
- 形状校验：每层权重形状必须完全匹配，否则加载报错

```python
def load_weights_into_gpt(model, params):
    # 词元嵌入
    model.tok_emb.weight = nn.Parameter(params['wte'])
    # 位置嵌入
    model.pos_emb.weight = nn.Parameter(params['wpe'])
    # 每个 Transformer 块
    for b in range(len(params['blocks'])):
        # Q/K/V 权重映射（注意 OpenAI 用 Conv1D，要转置）
        model.trf_blocks[b].att.W_query.weight = ...
        # 注意力输出投影
        # 前馈层
        # LayerNorm 的 scale/shift
    # 最终 LayerNorm
    # 输出头（权重共享时用 tok_emb 权重）
```

### 5.4 验证加载成功

加载后用 `generate` 生成文本，应输出连贯英文——这表明权重已正确加载。如果输出乱码，通常是配置/命名映射错位。

## 六、第 5 章小结

第 5 章完成了 **第二阶段：预训练 LLM**：

```
评估（5.1）：交叉熵/困惑度，训练集 vs 验证集损失
   ↓
训练（5.2）：train_model_simple，AdamW，8 步循环
   ↓
诊断过拟合（5.2）：train/val 损失发散 → 逐字记忆
   ↓
解码策略（5.3）：贪婪→温度→Top-k，减少记忆增加独创性
   ↓
权重管理（5.4）：torch.save / load，同时存 optimizer state
   ↓
加载 OpenAI 权重（5.5）：download_and_load_gpt2 + 配置对齐
   → 跳过昂贵预训练，直接做下游微调（第 6-7 章）
```

## 七、关键概念速查

| 术语 | 含义 |
|---|---|
| AdamW | Adam 变体，改进权重衰减，LLM 训练常用 |
| 过拟合 | train loss 下降但 val loss 停滞，模型死记训练集 |
| 贪婪解码 | 每步选最高概率词元，易逐字复现训练段落 |
| 温度缩放 | 用 τ 调节 softmax 尖锐度，τ>1 增多样性，τ<1 增确定性 |
| Top-k 采样 | 只从概率最高 k 个词元采样，避免低概率无意义词 |
| state_dict | PyTorch 模型参数字典 |
| 配置对齐 | 加载权重时必须匹配 context_length/qkv_bias/命名/形状 |
| 权重共享 | OpenAI GPT-2 把词元嵌入层当输出层复用 |

## 八、本篇要点

- 训练循环 8 步：遍历轮次→批次→zero_grad→前向→backward→step→监控→评估。
- AdamW 是 LLM 训练的常用优化器。
- 过拟合信号：train loss 下降但 val loss 停滞，且 train 远低于 val。诊断准则：看两曲线是否同步。
- 解码策略三档：贪婪（易复现）→ 温度缩放（调节多样性）→ Top-k（避免低概率词）。
- 保存权重时 **同时保存 optimizer state_dict**——否则续训会失去动量，模型可能不再生成连贯文本。
- 加载 OpenAI 权重要做配置对齐：context_length、qkv_bias、命名映射、形状校验——一处错配静默输出乱码。
- 加载成功后直接生成连贯文本，可跳过昂贵预训练直接微调（下两章）。
