# 08. GPT 模型架构组装

> 对应章节：第 4 章（4.1–4.6）

## 一、本章定位

第 2 章做完了数据预处理流水线（原始文本→训练批次→输入嵌入），第 3 章实现了多头注意力（LLM 核心组件）。第 4 章实现 LLM 的 **其他构建块**，并把它们组装成一个完整的 **类 GPT 模型**——在第 5 章训练它来生成类似人类语言的文本。

本章覆盖：

- 开发类 GPT 模型架构
- 层归一化稳定训练
- GELU 激活函数与前馈网络
- 快捷连接
- Transformer 块组装
- 计算参数量与存储需求

## 二、构建 GPT 架构的路线图

先自顶向下了解 GPT 数据流，再逐块实现：

```
词元ID → 词元嵌入 + 位置嵌入 → Dropout
       → Transformer块 × n_layers → LayerNorm
       → Linear (out_head) → logits
```

`DummyGPTModel` 占位架构先跑通流程，再逐个替换占位符为真实组件。

## 三、GPT-2 small 配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # BPE 词汇表大小
    "context_length": 1024, # 最大输入词元数
    "emb_dim": 768,         # 嵌入维度
    "n_heads": 12,          # 注意力头数
    "n_layers": 12,         # Transformer 块层数
    "drop_rate": 0.1,       # dropout 率
    "qkv_bias": False       # Q/K/V 是否带偏置
}
```

## 四、层归一化（LayerNorm）

### 4.1 为什么需要

训练深层神经网络会遇到 **梯度消失/梯度爆炸**——梯度在反向传播中逐渐变小或变大，导致早期层难以训练。

**层归一化** 调整神经网络层的激活（输出），使其均值为 0、方差为 1，加速收敛并稳定训练。

### 4.2 实现

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可训练缩放
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 可训练偏移
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

### 4.3 关键细节

- `eps`（epsilon）：小常数，加到方差上防止除零。
- `scale` 和 `shift`：可训练参数，让模型学习最佳缩放与偏移。
- `unbiased=False`：用 N（而非 N-1）作方差除数，不用贝塞尔修正——这是为了与原始 GPT-2（TensorFlow 实现）兼容，加载预训练权重时不至于错位。
- `dim=-1`：在嵌入维度上归一化，无论输入是 2 维还是 3 维都能处理。

### 4.4 LayerNorm vs BatchNorm

- **批归一化（BatchNorm）** 在批次维度上归一化，受批次大小影响。
- **层归一化（LayerNorm）** 在特征维度上归一化，**每个输入独立处理，不受批次大小限制**——这在分布式训练或资源受限部署时尤其重要。LLM 选 LayerNorm。

### 4.5 Pre-LayerNorm vs Post-LayerNorm

- **Pre-LayerNorm**（GPT 用）：LayerNorm 应用于自注意力和前馈层 **之前**。
- **Post-LayerNorm**（原始 Transformer 用）：在之后应用。
- 实践表明 Pre-LayerNorm 训练效果更好。

## 五、GELU 激活函数与前馈网络

### 5.1 GELU vs ReLU

历史上有 ReLU，但 LLM 常用更平滑的 **GELU（Gaussian Error Linear Unit）** 或 **SwiGLU**。

ReLU：分段线性，输入正则直接输出，否则输出 0。零点有尖锐拐角，深度网络优化难。

GELU：平滑非线性，对几乎所有负值都有非零梯度，让训练更易做细微调整。

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

（这是 GPT-2 用的近似实现，精确版本要用高斯累积分布函数，计算量更大。）

### 5.2 前馈网络 FeedForward

每个 Transformer 块内的子模块——两层 Linear + GELU：

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072 扩展
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # 3072 → 768 收缩
        )
    
    def forward(self, x):
        return self.layers(x)
```

**扩展-收缩设计**：先把嵌入维度从 768 扩展到 3072（4 倍），GELU 非线性变换，再缩回 768。这让模型探索更丰富的表示空间。输入输出维度一致，便于堆叠多层。

## 六、快捷连接（残差连接）

### 6.1 为什么需要

快捷连接（跳跃连接、残差连接）最初用于计算机视觉的 ResNet，目的是缓解 **梯度消失**——梯度从末层向前层传播时逐渐变小，导致早期层难训。

### 6.2 实现

快捷连接通过跳过一个或多个层，为梯度提供一条更短的替代路径——把一层的输出加到后续层的输出：

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 5 层，每层 Linear + GELU
    
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # 残差
            else:
                x = layer_output
        return x
```

### 6.3 实验对比

- 无快捷连接：5 层梯度从 0.005（末层）→ 0.0002（首层），梯度消失明显。
- 有快捷连接：5 层梯度从 1.32（末层）→ 0.22（首层），梯度保持稳定。

快捷连接是 LLM 等超大规模模型的核心构建块。

## 七、Transformer 块组装

把 MultiHeadAttention + FeedForward + LayerNorm + Dropout + 快捷连接组装成 **TransformerBlock**：

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # 子块 1：Pre-LN + 多头注意力 + 快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差
        
        # 子块 2：Pre-LN + 前馈 + 快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差
        return x
```

### 7.1 设计要点

- **多头注意力**：识别输入序列中元素间的关系（"看全局"）。
- **前馈网络**：在每个位置上单独修改数据（"做局部"）。
- 两者互补：先全局理解，再局部精修。
- **形状保持**：Transformer 块输入输出形状一致（如 `(batch, seq_len, 768)`），便于堆叠 n_layers 次。

## 八、GPT 模型完整实现

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

### 数据流

```
词元ID (b, seq_len)
   → 词元嵌入 + 位置嵌入 (b, seq_len, 768)
   → Dropout
   → TransformerBlock × 12
   → LayerNorm
   → Linear (768 → 50257)
   → logits (b, seq_len, 50257)
```

## 九、参数量与存储

### 9.1 实际参数量

```python
total_params = sum(p.numel() for p in model.parameters())
# 163,009,536  → 1.63 亿
```

但 GPT-2 标称是 1.24 亿——差在哪？

### 9.2 权重共享（weight tying）

原始 GPT-2 把 **词元嵌入层作为输出层重复使用**（权重共享）：

```python
# 词元嵌入层和输出层形状相同
# model.tok_emb.weight.shape == model.out_head.weight.shape == (50257, 768)
```

减去输出层参数：

```python
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# 124,412,160 → 1.24 亿
```

### 9.3 本书的取舍

权重共享能减少内存和计算量。但作者实验发现 **使用单独的词元嵌入层和输出层可获得更好的训练效果和模型性能**——本书 GPTModel 用单独的层。现代 LLM 也多采用单独层。

第 6 章加载 OpenAI 预训练权重时会再实现权重共享的概念，以兼容原始 GPT-2 权重。

### 9.4 存储需求

1.63 亿参数 × 4 字节（32 位浮点）≈ **621.83 MB**。即使是相对较小的 LLM 也需要相对较大的存储容量。

## 十、关键概念速查

| 术语 | 含义 |
|---|---|
| LayerNorm | 在嵌入维度归一化激活，使均值 0 方差 1，加速收敛 |
| Pre-LayerNorm | LayerNorm 在子模块之前应用（GPT 用，比 Post 更好） |
| GELU | 平滑激活函数，对负值有非零梯度，比 ReLU 优化更顺 |
| FeedForward | 前馈网络，768→3072→768 扩展-收缩设计 |
| 快捷连接 | 把一层输入加到后续层输出，为梯度提供短路径 |
| TransformerBlock | MultiHeadAttention + FeedForward + LayerNorm + 残差 + Dropout |
| 形状保持 | Transformer 块输入输出维度一致，便于堆叠 |
| 权重共享 | 原始 GPT-2 把词元嵌入当输出层复用，减参数；本书用单独层效果更好 |
| logits | 未经 softmax 的模型输出，每维对应词汇表一个词元 |

## 十一、本篇要点

- GPT = 词元嵌入 + 位置嵌入 + Dropout + N×TransformerBlock + LayerNorm + Linear 输出头。
- LayerNorm 在嵌入维度归一化，不受批次大小限制（优于 BatchNorm）；GPT 用 Pre-LayerNorm。
- GELU 是平滑激活，比 ReLU 更利于深度网络优化。
- FeedForward 用扩展-收缩设计（768→3072→768），输入输出维度一致便于堆叠。
- 快捷连接解决梯度消失——把输入加到输出，给梯度短路径。
- Transformer 块 = 多头注意力（看全局）+ 前馈（做局部）+ 残差 + Dropout + Pre-LN。
- GPTModel 参数量 1.63 亿（含输出层）/ 1.24 亿（权重共享），存储 621.83 MB。
- 至此第 4 章完成架构实现，下一章训练它。
