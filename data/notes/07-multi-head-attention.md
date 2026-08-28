# 07. 多头注意力机制实现

> 对应章节：第 3 章 3.6

## 一、为什么需要多头

### 1.1 单头的局限

上一篇实现的 CausalAttention 是一个"头"——它学一种注意力模式。但语言里词与词的关系是多维的：

- 语法关系（主谓宾）
- 语义关系（同义/反义）
- 共指关系（指代消解）
- 主题关系

一个头很难同时学好多维关系。**多头注意力（Multi-Head Attention）** 让模型并行运行多个注意力头，每个头学一种特征。

### 1.2 直觉

类比一组人读同一份文件：

- A 关注语法结构
- B 关注关键词重复
- C 关注指代关系
- ...

每人从不同角度看，最后汇总观点。多头注意力就是让模型这样做。

## 二、最朴素实现：实例化多个 CausalAttention

```python
batch = torch.stack((batched_input_1, batched_input_2))
context_length = batch.shape[1]
d_in = batch.shape[-1]
d_out = 2  # 教学用
num_heads = 6

heads = [CausalAttention(d_in, d_out, context_length, 0.0) for _ in range(num_heads)]
context_vecs = torch.cat([head(batch) for head in heads], dim=-1)
# 输出维度：num_heads × d_out = 6 × 2 = 12
```

问题：用 Python 列表 + 循环实现，效率低，且每个头是独立模块。

## 三、高效实现：MultiHeadAttention 类

把多个头封装成一个模块，用矩阵切片同时计算所有头：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 1. 计算 Q/K/V 并 reshape 为 (b, num_heads, num_tokens, head_dim)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. 缩放点积注意力（带掩码）
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. 加权和并 reshape 回 (b, num_tokens, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        
        # 4. 输出投影
        context_vec = self.out_proj(context_vec)
        return context_vec
```

### 关键技巧：reshape 与 transpose

- `W_query(x)` 输出形状 `(b, num_tokens, d_out)`。
- `.view(b, num_tokens, num_heads, head_dim)` 把最后一维拆成多头。
- `.transpose(1, 2)` 把头维度提到第二维，便于矩阵乘法并行处理所有头。

### out_proj 输出投影

多头拼接后加一个线性层 `out_proj`，让模型学习如何混合各头输出。原始 Transformer 也有这个投影层。

## 四、GPT-2 small 的注意力配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,        # 12 个注意力头
    "n_layers": 12,      # 12 层 Transformer 块
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

- 每个头维度：`768 / 12 = 64`。
- 12 个头并行，每个头学不同的注意力模式。
- 12 层 Transformer 块堆叠，每块都含一个 MultiHeadAttention。

## 五、注意力机制四步递进回顾

第 3 章的核心方法论是 **"每步只加一个新概念，每步可运行，输出差异是诊断信号"** 的可调试协议：

```
1. 简化自注意力（无可训练权重）
   局限：无法从数据学习
   ↓ 引入 Q/K/V 权重矩阵
2. 缩放点积注意力（带可训练权重）
   局限：会泄露未来位置
   ↓ 加上三角掩码
3. 因果注意力（带掩码）
   局限：只学一种特征
   ↓ 拆成多个头并行
4. 多头注意力（MultiHeadAttention）
   → 可嵌入 GPT 架构（第 4 章）
```

每一步都解决上一步的具体局限——这是 **用限制驱动设计** 的元方法论范本。

## 六、关键概念速查

| 术语 | 含义 |
|---|---|
| 多头注意力 | 多个注意力头并行运行，每头学一种特征 |
| head_dim | 每个头的维度 = d_out / num_heads |
| out_proj | 多头输出投影层，让模型学习如何混合各头输出 |
| register_buffer | 注册随模型迁移但不参与梯度的张量（如 mask） |
| view/transpose | reshape 技巧：把最后一维拆成多头，把头维度提到第二维 |
| n_heads | GPT-2 small 有 12 个头，每头 64 维 |

## 七、本篇要点

- 单头只学一种注意力模式，多头让模型并行学多维关系（语法/语义/共指/主题）。
- 高效实现用 reshape + transpose 把多头并行化，比朴素循环快得多。
- GPT-2 small：12 头 × 64 维 = 768 维输出，配 out_proj 投影。
- 注意力机制四步递进（简化→带权重→因果→多头）每步只补足上一步的具体局限，是"用限制驱动设计"的范本。
- 至此完成第 3 章——LLM 架构最关键的核心组件已就绪，下一章组装完整 GPT。
