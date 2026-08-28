# 12. 指令微调数据准备

> 对应章节：第 7 章 7.1–7.4

## 一、本章定位

第 6 章用分类微调把 GPT-2 变成"只能输出 0/1 的分类器"。第 7 章进入 **第三阶段第 (9) 步：微调 LLM 遵循人类指令**——让模型能聊天、回答问题、执行任务，而不只是分类。

这是开发聊天机器人、个人助理等对话应用的主要技术。

## 二、指令微调介绍

### 2.1 预训练模型的局限

预训练后的 LLM 能做文本补全——给定片段，生成句子或段落。但它在执行特定指令时表现不佳：

```
指令: "纠正这段文字的语法"
预训练模型输出:（继续续写指令，而非执行它）
```

base 模型只学了"下一词预测"，**不会遵循指令**——这是指令微调要解决的问题。

### 2.2 什么是指令微调

**指令微调（instruction fine-tuning）** 也叫 **有监督指令微调（supervised instruction fine-tuning）**：用结构化的"指令-回复"对训练模型，让它学会遵循人类指令并生成合理回复。

### 2.3 与分类微调的对比

| 维度 | 分类微调 | 指令微调 |
|---|---|---|
| 输出空间 | 锁死 N 维（类别） | 自由文本（词汇表） |
| 通用性 | 窄而深（专家） | 广而浅（通才） |
| 数据规模 | 少（百-千） | 多（千-万） |
| 计算资源 | 低 | 高 |
| 任务形态 | 封闭类别预测 | 开放指令响应 |

## 三、准备数据集

### 3.1 数据集来源

本书用一个专门创建的指令数据集，包含 **1100 个指令-回复对**，仅 204 KB（JSON 格式）。

```python
import json
with open("instruction-data.json", "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))  # 1100
```

### 3.2 样本结构

每个样本是包含 3 个键的字典：

```python
print(data[50])
# {'instruction': 'Identify the correct spelling of the following word.',
#  'input': 'Ocassion',
#  'output': "The correct spelling is 'Occasion.'"}

print(data[999])
# {'instruction': "What is an antonym of 'complicated'?",
#  'input': '',
#  'output': "An antonym of 'complicated' is 'simple'."}
```

- `instruction`：指令
- `input`：可选输入（可能为空）
- `output`：期望回复

### 3.3 划分数据集

```python
# 935 训练 / 55 验证 / 110 测试
train_portion = int(0.85 * len(data))   # 935
test_portion = int(0.10 * len(data))    # 110
val_portion = len(data) - train_portion - test_portion  # 55
```

## 四、提示词风格选择

### 4.1 Alpaca vs Phi-3

指令微调需要把样本格式化成模型能理解的提示词。常见的两种风格：

**Alpaca 风格**（结构化）：
```
Below is an instruction that describes a task. Write a response
that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**Phi-3 风格**（更简洁，用特殊词元 `<|user|>` `<|assistant|>`）：
```
<|user|>
{instruction}
<|assistant|>
{output}
```

### 4.2 本书选 Alpaca

Alpaca 是最早公开详细说明其指令微调过程的 LLM 之一，奠定了指令微调的提示词风格基础。本书默认用 Alpaca 风格。

### 4.3 format_input 函数

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text
```

注意：如果 `entry['input']` 为空，跳过 `### Input:` 小节。

```python
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
```

## 五、将数据组织成训练批次

### 5.1 为什么需要自定义聚合函数

第 6 章用 `DataLoader` 默认聚合函数，但 **指令微调的批次处理更复杂**——需要创建一个自定义 `collate_fn`（聚合函数）来满足指令微调的特定需求和格式。

### 5.2 五子步骤

```
(2.1) 应用提示词模板（format_input）
(2.2) 词元化（tiktoken gpt2）
(2.3) 添加填充词元（50256）
(2.4) 创建目标词元ID（输入向左移一位 + 末尾加 50256 结束符）
(2.5) 在损失中用 -100 占位符掩码填充词元
```

### 5.3 InstructionDataset 类（步骤 2.1 + 2.2）

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = [
            tokenizer.encode(format_input(entry)) for entry in data
        ]
        # 预词元化（在 __init__ 一次性做完，训练时只查表）
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
```

### 5.4 自定义聚合函数（步骤 2.3-2.5）

**步骤 2.3：批次内填充**——每个批次填到该批次最长，**不同批次长度可不同**（61/76/73 等），减少不必要填充。

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=1024):
    batch_max_len = max(len(item) for item in batch)
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        # 步骤 2.3 填充输入
        new_item = item + [pad_token_id] * (batch_max_len - len(item))
        # 步骤 2.4 创建目标：输入左移一位 + 末尾加 50256
        # 注意：目标的第一个元素是输入的第二个，最后一个加结束符
        # 步骤 2.5 把填充词元 50256 替换为 -100，但保留一个真实 50256
        inputs_lst.append(new_item[:-1])    # 去掉最后一个作为输入
        targets_lst.append(new_item[1:])     # 去掉第一个作为目标
    
    inputs = torch.tensor(inputs_lst)
    targets = torch.tensor(targets_lst)
    # 把 targets 中的 50256（除第一个）替换为 -100
    ...
    return inputs, targets
```

### 5.5 关键细节：-100 占位符与保留结束符

**为什么要 -100？**

PyTorch 的 `cross_entropy` 默认 `ignore_index=-100`——标记为 -100 的目标会被忽略，不参与损失计算。我们用 -100 掩码填充词元，让损失只算有效数据。

**为什么保留一个 50256？**

目标中保留一个真实的 `模型学会何时生成结束符，从而在适当时候结束回复——否则微调后模型会一直生成不停。

```
目标示例:
[词1, 词2, 词3, 50256, -100, -100, -100]
                ↑ 保留       ↑ 掩码（填充）
```

### 5.6 验证 -100 行为

```python
# 不带 -100
logits_2 = torch.tensor([[-1.0, 1.0], [-3.0, 6.0], [1.0, -1.0]])
targets_2 = torch.tensor([0, 1, 1])
loss = cross_entropy(logits_2, targets_2)  # tensor(0.7936)

# 把第 3 个目标改为 -100
targets_3 = torch.tensor([0, 1, -100])
loss = cross_entropy(logits_2, targets_3)  # tensor(1.1269) = 不算第 3 个
```

-100 让交叉熵忽略对应位置，等价于该位置不参与训练。

## 六、创建数据加载器

### 6.1 用 partial 预填设备参数

```python
from functools import partial
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_loader = DataLoader(
    InstructionDataset(train_data, tokenizer),
    batch_size=8,
    shuffle=True,
    collate_fn=customized_collate_fn,
    drop_last=True
)
```

### 6.2 不同批次不同长度

```python
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
# torch.Size([8, 61]) torch.Size([8, 61])
# torch.Size([8, 76]) torch.Size([8, 76])
# torch.Size([8, 73]) torch.Size([8, 73])
# ...
```

批次大小 8，每个批次的序列长度不同（61/76/73 等）——这是动态填充的优势。

### 6.3 设备设置放在 collate_fn

把数据搬到 GPU/Apple Silicon 的操作写在聚合函数里，可在训练循环外后台执行，避免训练时阻塞 GPU。

## 七、关键决策回顾

| 决策点 | 选择 | 理由 |
|---|---|---|
| 提示词风格 | Alpaca | 奠基风格，公开说明详细 |
| 批次填充 | 批次内最长，不同批次可不同 | 减少不必要填充 |
| 目标构造 | 输入左移一位 + 末尾加 50256 | 标准自回归训练目标 |
| 填充词元掩码 | -100 | PyTorch cross_entropy 默认 ignore_index |
| 保留结束符 | 保留一个 50256 | 让模型学会结束回复 |
| 数据搬到设备 | 在 collate_fn 里做 | 后台执行，不阻塞训练 |

## 八、本篇要点

- 预训练模型不会遵循指令——base 模型只学下一词预测，会把指令当文本续写。指令微调解决此问题。
- 指令微调与分类微调对比：输出自由文本（vs 锁死 N 维）、广而浅（vs 窄而深）、需更多数据/算力。
- 数据集 1100 个 instruction-input-output 对，按 935/55/110 划分。
- 提示词风格选 Alpaca（结构化 `### Instruction/### Input/### Response`），Phi-3 是另一种风格。
- 自定义 collate_fn 五步：模板→词元化→填充→目标构造→-100 掩码。
- **批次内动态填充**：每个批次填到该批次最长，不同批次可不同长度——减少不必要填充。
- **-100 掩码填充词元**：PyTorch `cross_entropy(ignore_index=-100)` 默认忽略 -100，让损失只算有效数据。
- **保留一个真实结束符 50256**：让模型学会何时结束回复，否则微调后会一直生成不停。
- 把数据搬 GPU 写在 collate_fn 里，可后台执行不阻塞训练。
- 下篇加载 gpt2-medium (355M) 并微调。
