# 14. PyTorch 基础与开发环境

> 对应章节：附录 A 核心 + 开发环境配置

## 一、为什么需要 PyTorch

本书所有代码示例用 PyTorch 实现。PyTorch 是当前深度学习主流框架之一，由 Meta（Facebook）AI 研究团队开发。本书把它作为从零实现 LLM 的主要张量和深度学习库。

如果你对 PyTorch 不熟悉，建议先读附录 A 再开始第 2 章。本篇提炼附录 A 的核心内容。

## 二、什么是张量

### 2.1 张量是 LLM 的"数"

张量（tensor）是 PyTorch 的核心数据结构——多维数组。LLM 里的所有数据（词元嵌入、注意力分数、梯度、权重）都是张量。

| 维度 | 名称 | 例子 |
|---|---|---|
| 0 维 | 标量 | `tensor(3.14)` |
| 1 维 | 向量 | `tensor([1.0, 2.0, 3.0])` |
| 2 维 | 矩阵 | `tensor([[1, 2], [3, 4]])` |
| 3 维 | 张量 | 嵌入批次 `(batch, seq, emb)` |
| N 维 | N 阶张量 | LLM 内部到处都是 |

### 2.2 张量的关键属性

```python
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(x.shape)     # torch.Size([2, 2])  形状
print(x.dtype)      # torch.float32       数据类型
print(x.device)     # cpu 或 cuda         所在设备
print(x.requires_grad)  # 是否需要梯度
```

### 2.3 维度操作

- `dim=0`：沿行方向（垂直），结果对每列汇总。
- `dim=1` 或 `dim=-1`：沿列方向（水平），结果对每行汇总。
- `keepdim=True`：保持输出维度与输入一致（不挤压维度）。

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x.mean(dim=-1, keepdim=True)  # [[2.], [5.]]  每行均值
```

LLM 处理 3 维张量 `(batch, seq, emb)` 时，常用 `dim=-1` 在嵌入维度上操作（如 LayerNorm）。

## 三、自动微分（autograd）

### 3.1 PyTorch 的核心能力

PyTorch 能 **自动计算梯度**——这是它能训练神经网络的关键。你只需要：

1. 前向计算 loss
2. 调用 `loss.backward()`，PyTorch 自动算出所有参数的梯度
3. 调用 `optimizer.step()` 更新参数

### 3.2 计算图

PyTorch 在前向传播时自动构建 **计算图**——记录每一步运算。反向传播时沿图反向算梯度。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3
y.backward()
print(x.grad)  # tensor(4.) = dy/dx = 2x = 4
```

### 3.3 requires_grad 的开关

- 训练时：所有需要学习的参数 `requires_grad=True`。
- 推理时：用 `torch.no_grad()` 或 `model.eval()` 关闭——节省内存和算力。
- 微调时冻结底层：`param.requires_grad = False`。

```python
# 推理
with torch.no_grad():
    outputs = model(inputs)

# 冻结某些层
for param in model.parameters():
    param.requires_grad = False
```

## 四、nn.Module 与构建模型

### 4.1 所有模型都是 nn.Module 的子类

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
```

### 4.2 关键方法

| 方法 | 作用 |
|---|---|
| `model.parameters()` | 返回所有可训练参数 |
| `model.named_parameters()` | 返回参数名+参数 |
| `model.state_dict()` | 返回参数字典（保存/加载用） |
| `model.eval()` | 切到评估模式（关闭 Dropout） |
| `model.train()` | 切回训练模式 |
| `model.to(device)` | 搬到 GPU/CPU |

### 4.3 nn.Linear、nn.Embedding、nn.Dropout、nn.Sequential

- `nn.Linear(in, out)`：线性层 y = Wx + b
- `nn.Embedding(vocab, dim)`：查表，把整数 ID 转为向量
- `nn.Dropout(p)`：训练时随机置零，推理时不变
- `nn.Sequential(...)`：把多个层串成序列

本书的 GPTModel 就是这些组件的组合：

```python
self.trf_blocks = nn.Sequential(
    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
)
```

## 五、训练循环详解

### 5.1 八步流程

```python
for epoch in range(num_epochs):           # 1. 遍历轮次
    model.train()                         # 切训练模式
    for inputs, targets in train_loader:  # 2. 遍历批次
        optimizer.zero_grad()             # 3. 重置梯度
        outputs = model(inputs)           # 4. 前向
        loss = loss_fn(outputs, targets)  #    计算损失
        loss.backward()                   # 5. 反向
        optimizer.step()                  # 6. 更新权重
        
        if step % eval_freq == 0:         # 7. 定期评估
            train_loss = evaluate_model(model, train_loader)
            val_loss = evaluate_model(model, val_loader)
```

### 5.2 为什么每次都要 zero_grad

PyTorch 默认 **累积梯度**——不调用 `zero_grad()` 会让新梯度叠到旧梯度上，导致训练错乱。这是初学者最常犯的 bug。

### 5.3 model.eval() 与 no_grad 的区别

- `model.eval()`：切评估模式，影响 **Dropout 和 BatchNorm**（Dropout 关闭，BatchNorm 用历史统计量）。但**不会关闭梯度计算**。
- `torch.no_grad()`：**关闭梯度计算**，节省内存和算力。

推理时应该 **同时用**：

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

### 5.4 优化器

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
```

- `lr`（学习率）：步长，太大震荡，太小慢。
- `weight_decay`（权重衰减）：L2 正则，惩罚大权重，防过拟合。

学习率调度：附录 D 介绍高级技巧如学习率预热、余弦衰减、梯度裁剪。

## 六、Dataset 与 DataLoader

### 6.1 Dataset

```python
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### 6.2 DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,        # 训练时打乱
    drop_last=True,      # 丢弃最后不完整批次
    collate_fn=custom_fn, # 自定义聚合（指令微调用）
    num_workers=0        # 并行加载
)
```

### 6.3 collate_fn（聚合函数）

`DataLoader` 默认把样本列表堆叠成张量批次。但 **指令微调需要自定义聚合**——做动态填充、目标构造、-100 掩码（详见第 12 篇）。

## 七、设备管理

### 7.1 三种设备

- `cpu`：通用，慢
- `cuda`：NVIDIA GPU，快
- `mps`：Apple Silicon（M1/M2/M3 Mac），实验性支持

### 7.2 自动选择设备

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### 7.3 数据与模型都要搬设备

```python
model = model.to(device)
inputs = inputs.to(device)
```

注意：把数据搬设备写在 collate_fn 里可后台执行，避免训练时阻塞 GPU（第 12 篇）。

### 7.4 Apple Silicon 注意事项

用 `mps` 设备可能导致数值结果与书中略有差异——PyTorch 对 Apple Silicon 的支持仍处于实验阶段。

## 八、保存与加载

### 8.1 保存

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pth")
```

**关键：同时保存 model 和 optimizer 的 state_dict**——optimizer 内部有动量等状态，不保存会让续训失去连贯性，模型可能失去生成连贯文本的能力。

### 8.2 加载

```python
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

## 九、开发环境配置

### 9.1 必备库

```bash
pip install torch torchvision torchaudio
pip install tiktoken       # BPE 分词器
pip install matplotlib     # 画损失曲线
pip install numpy pandas   # 数据处理
pip install tqdm           # 进度条
```

### 9.2 本书的代码组织

本书代码按章节分文件夹：

```
LLMs-from-scratch/
├── ch02/  # 数据处理
├── ch03/  # 注意力机制
├── ch04/  # GPT 架构
├── ch05/  # 预训练
├── ch06/  # 分类微调
├── ch07/  # 指令微调
└── appendix-A/  # PyTorch 入门
```

每章的 `chapterXX.py` 可被后续章节 `from chapterXX import ...` 复用。

### 9.3 Jupyter Notebook

GitHub 上的代码以 Jupyter Notebook 格式提供，便于交互式学习与可视化。

## 十、关键概念速查

| 术语 | 含义 |
|---|---|
| 张量 | 多维数组，LLM 所有数据的形式 |
| dim/keepdim | 操作维度 / 是否保持维度 |
| autograd | 自动微分，前向构建图、反向算梯度 |
| requires_grad | 是否需要梯度；冻结层置 False |
| nn.Module | 所有模型基类 |
| state_dict | 参数字典，保存/加载用 |
| model.eval() | 切评估模式（关 Dropout），不关梯度 |
| torch.no_grad() | 关梯度计算，省内存 |
| zero_grad | 重置梯度（PyTorch 默认累积） |
| AdamW | 带权重衰减的 Adam，LLM 常用 |
| collate_fn | 自定义批次聚合函数（指令微调用） |
| device | cpu / cuda / mps |

## 十一、本篇要点

- PyTorch 是本书实现 LLM 的主框架，张量是其核心数据结构。
- `dim=-1` 在嵌入维度操作，是 LLM 常用维度处理方式。
- autograd 自动算梯度——你只需 `loss.backward()` + `optimizer.step()`。
- `model.eval()` 关 Dropout，`torch.no_grad()` 关梯度——推理时同时用。
- 每次 step 前必须 `zero_grad()`——PyTorch 默认累积梯度，不重置会训练错乱。
- Dataset + DataLoader 是数据加载标准范式；指令微调需要自定义 collate_fn 做动态填充。
- 保存 checkpoint 时 **同时保存 optimizer state**，否则续训失去动量。
- Apple Silicon 用 `mps` 设备，是实验性支持，数值可能有细微差异。
- 必备库：torch / tiktoken / matplotlib / numpy / pandas / tqdm。
