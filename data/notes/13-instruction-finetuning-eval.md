# 13. 指令微调训练与 LLM-as-judge 评估

> 对应章节：第 7 章 7.5–7.8

## 一、加载预训练模型

### 1.1 选 gpt2-medium (355M) 而非 small (124M)

本节加载预训练 GPT 模型并准备微调。但这次 **不再用 124M 的最小模型，而是用 3.55 亿的中等规模模型**：

```python
model_configs = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"
```

### 1.2 为什么不用 124M

**124M 容量过于有限，无法通过指令微调获得令人满意的结果**——较小的模型在学习高质量的指令遵循任务时，缺乏执行该任务所需的复杂模式和细微行为的能力。

这是一个常见误判：很多人以为"小模型是大模型等比缩小，调参就行"——实际是指令遵循有 **容量门槛**，124M 调参也学不会。

### 1.3 配置变更

加载 gpt2-medium 时要调整配置：

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True  # ★ 注意改为 True，与 OpenAI 权重匹配
}
BASE_CONFIG.update(model_configs["gpt2-medium (355M)"])
# emb_dim=1024, n_layers=24, n_heads=16
```

`qkv_bias=True` 是关键——与第 6 章加载权重时一致，否则权重加载会错位。

### 1.4 下载与加载

```python
settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

gpt2-medium 约 1.42 GB，是最小 GPT 的 3 倍存储。

### 1.5 微调前的基线测试

用验证集第一个样本评估预训练模型：

```
输入: "Convert the active sentence to passive: 'The chef cooks the meal every day.'"
模型输出:（模型续写指令而非执行，输出不连贯）
```

这是预期——预训练模型缺乏指令遵循能力。微调后会改善。

## 二、微调训练

### 2.1 训练配置

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 2
```

- **学习率 5e-5**：比预训练（4e-4）更小——微调只调整已有权重，避免破坏预训练知识。
- **2 轮**：指令微调数据集小（935 训练样本），多轮易过拟合。

### 2.2 训练循环

与第 5 章预训练类似，但用第 12 章的 `custom_collate_fn` 和指令数据集：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(inputs, targets, model, device)
        loss.backward()
        optimizer.step()
    
    # 评估
    train_loss = evaluate_model(model, train_loader, device)
    val_loss = evaluate_model(model, val_loader, device)
```

### 2.3 微调轮数：5 轮起点

**5 轮是微调训练的不错起点**——这是经验值。但本书指令数据集小，2 轮就能达到不错效果。延长训练轮数反而会过拟合加剧，"多训几轮"不是万能解。

### 2.4 训练结果

微调 2 轮后，验证损失明显下降，模型从"续写指令"变为"生成回复"：

```
微调前: "Convert the active sentence to passive: 'The chef cooks the meal every day.' The meal is cooked by the chef every day is..."
微调后: "The meal is cooked every day by the chef."
```

## 三、提取指令响应

### 3.1 为什么需要提取

微调后模型会生成完整文本，包含指令部分和回复部分。**评估时只需要回复部分**——所以要从生成文本中提取出 `### Response:` 后面的内容。

### 3.2 generate_and_extract

```python
def generate_response(model, tokenizer, instruction_input, max_new_tokens=50, context_length=1024):
    model.eval()
    input_text = format_input(entry)  # Alpaca 格式
    input_ids = tokenizer.encode(input_text)
    
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=torch.tensor([input_ids]),
            max_new_tokens=max_new_tokens,
            context_size=context_length
        )
    
    response_text = tokenizer.decode(token_ids[0][len(input_ids):])
    # 从生成文本中提取 ### Response: 之后的部分
    response = response_text.split("### Response:")[1].strip()
    response = response.split("### Instruction:")[0].strip()  # 截断后续生成
    return response
```

### 3.3 处理多轮生成的截断

模型可能不会自动停止——会一直生成直到 `max_new_tokens`。提取时要在合理位置截断（如遇到下一个 `### Instruction:` 或结束符）。

## 四、LLM-as-judge 自动评估

### 4.1 评估难点

指令微调后回复质量是 **多维主观的**：

- 是否准确回答了指令
- 是否流畅
- 是否安全
- 是否简洁

单标量损失（交叉熵）不够——它只能反映"模型预测与目标词元的差异"，不能反映"回复是否有用"。人工评估 110 条测试样本可行，但 1 万条就不可扩展。

### 4.2 LLM-as-judge 思路

**用更强的 LLM 当评分员**，把主观判断工程化为可批量复现的数值指标。

本书用本地跑的 Llama 3-8B（通过 Ollama）作为评分员：

```python
# 评分 prompt（给评分 LLM 的）
prompt = f"""
Given the instruction:
{instruction}

And the correct response:
{correct_response}

And the model response:
{model_response}

Rate the model response on a scale of 0-100.
"""
```

### 4.3 改提示词为"只返回整数"

为批量自动统计，把评分 prompt 改为让评分 LLM 只返回整数分数：

```python
prompt = f"""
You will be given an instruction, a correct response, and a model response.
Rate the model response as integer 0-100.
Return only the integer, nothing else.
"""
```

这样可对 110 条测试样本批量跑，算平均分。

### 4.4 结果对比

```
微调前（base 模型）平均分: 约 30-50
微调后（指令微调）平均分: 约 70-90
```

### 4.5 LLM-as-judge 的权衡

- **优点**：成本低，可批量，可复现，比纯人工评估可扩展。
- **缺点**：评估质量受评分模型上限制约——是人工评估的低成本近似，不是银弹。
- **不适用场景**：分类任务（直接用准确率）、有标准答案的客观题、对评估精度要求极高且预算允许人工的场景。

## 五、综合验证：微调前 vs 微调后

### 5.1 测试集 110 样本对比

| 指标 | 微调前 | 微调后 |
|---|---|---|
| 交叉熵损失 | 高 | 低 |
| LLM-as-judge 平均分 | 30-50 | 70-90 |
| 回复是否回答指令 | 否（续写指令） | 是 |
| 回复是否流畅 | 部分 | 是 |

### 5.2 抽样对比

```
指令: "Convert the active sentence to passive: 'The chef cooks the meal every day.'"

微调前（base 模型）:
"The meal is cooked by the chef every day is cooking the meal is the meal
is cooked by the chef every day the meal is cooked..."

微调后（指令微调）:
"The meal is cooked every day by the chef."
```

## 六、第 7 章关键决策回顾

| 决策点 | 选择 | 理由 |
|---|---|---|
| 模型规模 | gpt2-medium (355M) | 124M 容量不足，指令微调学不会 |
| qkv_bias | True | 与 OpenAI 权重匹配 |
| 学习率 | 5e-5 | 比预训练小，避免破坏预训练知识 |
| 训练轮数 | 2 轮 | 数据集小，多轮易过拟合；5 轮是通用起点 |
| 评估方法 | LLM-as-judge | 主观多维质量可批量复现，比人工可扩展 |
| 评分模型 | Llama 3-8B | 本地跑，无 API 成本 |
| 评分输出 | 只返回整数 | 便于批量统计 |

## 七、本篇要点

- **指令微调有容量门槛**——124M 调参也学不会，要用 gpt2-medium (355M) 或更大。这是常见误判。
- 加载 gpt2-medium 时 `qkv_bias=True`，与 OpenAI 权重匹配。
- 微调学习率 5e-5（比预训练 4e-4 小），避免破坏预训练知识。
- 微调轮数 2 轮（小数据集），通用起点 5 轮；延长训练反增过拟合。
- 评估指令微调模型不能用单标量损失——回复质量多维主观。
- **LLM-as-judge**：用更强 LLM 当评分员，把主观质量工程化为可批量复现的整数指标。
- 评分 prompt 改为"只返回整数"以便批量统计平均分。
- LLM-as-judge 是人工评估的低成本近似，不是银弹；不适用于分类等有标准答案的任务。
- 微调后模型从"续写指令"变为"生成回复"，LLM-as-judge 平均分从 30-50 升到 70-90。
