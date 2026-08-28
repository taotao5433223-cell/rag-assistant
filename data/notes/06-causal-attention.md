# 06. 带可训练权重的自注意力与因果掩码

> 对应章节：第 3 章 3.4–3.5

## 一、从简化版到缩放点积注意力

上一篇实现的简化自注意力没有可训练参数，无法从数据中学习"应该关注谁"。本节加入 **可训练权重矩阵**，实现原始 Transformer、GPT 和大多数流行 LLM 使用的 **缩放点积注意力（scaled dot-product attention）**。

### 1.1 三个权重矩阵：Q / K / V

引入 3 个可训练权重矩阵 W_Q、W_K、W_V，将嵌入的输入词元分别映射为：

- **查询向量（Query, Q）**：当前位置"我想查什么"
- **键向量（Key, K）**：每个位置"我能被什么匹配"
- **值向量（Value, V）**：每个位置"我提供的信息内容"

```python
# 输入 x² 与权重矩阵 W_Q 相乘得查询向量 q²
# 同样地，所有输入与 W_K、W_V 相乘得键向量和值向量
```

类比图书馆检索：Q 是你的检索词，K 是每本书的标题/关键词，V 是每本书的内容。检索词与某本书标题越匹配（点积越大），你越关注那本书的内容。

### 1.2 计算流程（带可训练权重）

```
1. 输入 x^i 与 W_Q、W_K、W_V 相乘 → 得 q^i、k^i、v^i
2. 注意力分数 ω_ij = q^i · k^j  （查询与键的点积）
3. 注意力权重 α_ij = softmax(ω_ij / sqrt(d_k))  ← 这就是"缩放"
4. 上下文向量 z^i = Σ_j α_ij · v^j  （值向量的加权和）
```

### 1.3 为什么除以 sqrt(d_k)（缩放）

点积的大小会随嵌入维度 d_k 增长。如果不缩放，点积可能变得很大，使 softmax 进入饱和区——梯度趋近 0（梯度消失），训练停滞。除以 `sqrt(d_k)` 把分数控制回合理范围，保持梯度健康。

这就是 "scaled" dot-product attention 中"缩放"的来源，也是原始 Transformer 论文的做法。

### 1.4 单条上下文向量的逐步代码

```python
# 1. 计算 Q、K、V
queries = inputs @ W_Q       # (6, emb_dim) @ (emb_dim, emb_dim) → (6, out_dim)
keys = inputs @ W_K
values = inputs @ W_V

# 2. 注意力分数（查询与键的点积）
attn_scores = queries @ keys.T   # (6, 6)

# 3. 缩放并 softmax
d_k = keys.shape[-1]
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

# 4. 上下文向量（值向量加权和）
context_vecs = attn_weights @ values
```

### 1.5 SelfAttention 类封装

把上述步骤封装成 PyTorch 模块，便于在第 4 章嵌入 GPT 架构：

```python
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
```

`nn.Linear` 默认含偏置（bias）。本书的 GPT 在加载 OpenAI 权重时需匹配 `qkv_bias` 设置（详见第 6 章）。

## 二、因果注意力（掩码自注意力）

### 2.1 为什么需要因果掩码

简化自注意力让每个位置能访问所有位置（包括未来位置）。但 GPT 是 **自回归生成** 模型——逐词预测下一个词，**不能让当前位置看到未来位置**（否则训练时就在作弊）。

**因果注意力（causal attention）** = 在自注意力上加掩码，使位置 i 只能关注位置 0..i。

### 2.2 用上三角掩码实现

```python
# 1. 计算注意力分数
attn_scores = queries @ keys.T

# 2. 创建上三角掩码（对角线以上置 -inf）
batch_size, context_size = attn_scores.shape
mask = torch.triu(torch.ones(context_size, context_size), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# 3. softmax（-inf 位置归一化后变为 0）
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

掩码后的注意力权重矩阵是下三角形式：每行从位置 0 加到当前位置，未来位置权重为 0。

### 2.3 直觉：因果掩码不是"信息泄露"

担心"掩码后位置 i 仍能间接通过位置 i-1 知道未来信息"是一个直觉陷阱。实际上每个位置的上下文向量只用了历史位置的值向量，没有泄露未来。

## 三、Dropout 正则化

### 3.1 为什么需要 Dropout

为减少过拟合，在注意力权重上加 **Dropout**：训练时随机把部分权重置 0（按概率 `drop_rate`，如 0.1 表示 10% 单元被丢弃）。

### 3.2 Dropout 的使用要点

- **仅训练时启用**，推理时关闭（`model.eval()` 切换）。
- 在 GPT 中，Dropout 应用于：注意力权重之后、快捷连接之后、嵌入之后。
- 本书的 GPT-2 配置 `drop_rate = 0.1`。

```python
dropout = torch.nn.Dropout(p=0.5)  # 教学用，实际 0.1
attn_weights = dropout(attn_weights)
```

推理时忘记关闭 Dropout 是常见 bug——会随机丢弃已学到的信息，导致输出异常。

## 四、把多步封装为 CausalAttention 类

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_size), diagonal=1))
    
    def forward(self, x):
        # x 形状: (batch, seq_len, d_in)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = attn_scores.masked_fill(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
```

注意 `register_buffer` 注册的 mask 会随模型一起搬到 GPU/移动设备，且不会参与梯度更新。

## 五、本篇要点

- 缩放点积注意力 = 简化自注意力 + 可训练的 Q/K/V 权重矩阵。
- Q 是查询，K 是键，V 是值——类比图书馆检索（检索词/标题/内容）。
- **除以 sqrt(d_k)** 是"缩放"的来源——防止点积过大使 softmax 饱和、梯度消失。
- 因果注意力用上三角掩码（-inf），让位置 i 只能看 0..i，支持自回归生成。
- 因果掩码不会导致信息泄露，是直觉陷阱。
- Dropout 仅训练时启用，推理必须 `model.eval()`——否则会随机丢弃已学信息。
- `register_buffer` 注册的掩码随模型迁移，不参与梯度。

下一篇我们把多个 CausalAttention 头并行堆叠成 **多头注意力**——这是注意力机制四步递进的最后一步。
