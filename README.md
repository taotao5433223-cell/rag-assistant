# RAG 问答助手

基于《从零构建大模型》15 篇读书笔记，**从零手写** RAG（检索增强生成）全流程的智能问答系统。

不使用 LangChain 等框架，用 Python 原生实现切分、向量化、检索、生成的每一步——**深入理解 RAG 全流程原理与实现细节**。

## 功能特性

- **按标题结构化切分**：识别 Markdown `##`/`###` 标题边界，避免切断代码块
- **语义检索**：手写余弦相似度，从向量库中召回 Top-K 最相关文档块
- **幻觉抑制**：LLM 只依据检索到的资料回答，资料中没有的内容回复"资料中没有相关内容"
- **来源溯源**：回答末尾标注引用的文档来源
- **网页界面**：Gradio 搭建的聊天式交互界面，支持示例问题一键点击

## 技术栈

| 环节 | 技术 | 说明 |
|------|------|------|
| 知识源 | 15 篇 Markdown 读书笔记 | 覆盖 LLM 全链路：Transformer/GPT/注意力/BPE/预训练/微调等 |
| 文档切分 | Python 手写 | 按 `##`/`###` 标题切分 + 超长块补切 + 代码块保护 |
| 向量化 | DashScope `text-embedding-v4` | 阿里云百炼，1024 维 |
| 向量检索 | 手写余弦相似度 | 纯 Python `math` + `zip`，无向量数据库依赖 |
| 答案生成 | GLM-5.3-Flash（智谱） | 通过 OpenAI 兼容接口调用 |
| 网页界面 | Gradio 6 | `ChatInterface` 聊天界面 + 自定义主题样式 |

## 项目结构

```
rag-assistant/
├── data/
│   ├── notes/                  # 15 篇 Markdown 读书笔记（知识源）
│   │   ├── 01-llm-overview-and-learning-path.md
│   │   ├── 02-transformer-and-gpt-architecture.md
│   │   ├── 03-embedding-and-tokenization.md
│   │   ├── 04-bpe-and-sliding-window.md
│   │   ├── 05-attention-basics.md
│   │   ├── 06-causal-attention.md
│   │   ├── 07-multi-head-attention.md
│   │   ├── 08-gpt-architecture.md
│   │   ├── 09-pretraining-evaluation.md
│   │   ├── 10-training-loop-and-weights.md
│   │   ├── 11-classification-finetuning.md
│   │   ├── 12-instruction-data-prep.md
│   │   ├── 13-instruction-finetuning-eval.md
│   │   ├── 14-pytorch-basics.md
│   │   └── 15-full-pipeline-summary.md
│   ├── chunks.json              # 切分后的文本块（由 chunk.py 生成）
│   └── embeddings.json          # 向量化后的数据（由 embed.py 生成）
├── src/
│   ├── config.py                # API 密钥配置（已在 .gitignore 中）
│   ├── chunk.py                 # 文档切分
│   ├── embed.py                 # 向量化
│   ├── vector_store.py          # 手写余弦相似度检索
│   ├── generate.py              # 检索 + 拼接 Prompt + 调用 LLM
│   └── app.py                   # Gradio 网页界面
├── .gitignore
└── README.md
```

## RAG 五步流程

本项目完整实现了 RAG 的五个核心步骤，每一步都是手写代码，不套框架：

```
[离线·建库阶段]
Markdown 笔记 → ①切分 → ②向量化 → ③存入 embeddings.json

[在线·问答阶段]
用户提问 → ②向量化(同一个模型) → ④余弦相似度检索Top-K → ⑤拼Prompt调LLM → 回答+来源
```

### ① 文档切分（`chunk.py`）

将 15 篇 Markdown 笔记按 `##`/`###` 标题切分为独立小节，每个小节是一个完整的知识单元。

核心逻辑：
- `split_sections()`：按标题切分，跟踪代码块状态（`in_code`），避免把代码块内的 `###` 误当标题
- `make_chunks()`：超长小节按换行符补切，且检查代码围栏平衡（` ``` ` 奇数时移动切点到闭合围栏之后），避免切断代码块
- 过滤空壳小节（只有标题无正文，`len < 30` 跳过）

```bash
python src/chunk.py
# 输出 data/chunks.json，约 296 个文本块
```

### ② 向量化（`embed.py`）

调用 DashScope `text-embedding-v4` 接口，把每个文本块转为 1024 维浮点向量。

- 批量逐块调用，含 3 次重试机制（指数退避）
- 控频 `sleep(1)` 避免触发 API 限流
- 结果存入 `data/embeddings.json`，每条含 `{file, text, vector}`

```bash
python src/embed.py
# 输出 data/embeddings.json，约 296 条向量数据
```

### ③ 存储

使用 JSON 文件存储向量数据（`embeddings.json`），无需部署向量数据库。每条记录结构：

```json
{
  "file": "05-attention-basics.md",
  "text": "三、注意力机制捕捉数据依赖\n## 三、...",
  "vector": [0.012, -0.435, 0.877, ...]
}
```

### ④ 检索（`vector_store.py`）

手写余弦相似度，计算用户问题向量与所有文档块向量的相似度，取 Top-K：

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0

def search(query_vector, records, top_k=3):
    scored = [(cosine(query_vector, r["vector"]), r) for r in records]
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]
```

- 余弦相似度衡量向量**方向**的接近程度，不受文本长度影响
- `sort(key=lambda x: -x[0])` 按相似度降序排列，取最相关的前 3 条

### ⑤ 生成（`generate.py`）

将检索到的 Top-K 文档块拼入 Prompt，调用 GLM-5.3-Flash 生成回答：

```
系统指令：你只能根据下面的资料回答，资料里没有的内容，回答"资料中没有相关内容"。
资料：[检索到的文档块]
问题：[用户输入]
```

- `temperature=0`：确定性输出，减少幻觉
- 约束 LLM 只用检索资料回答，无资料时明确回复"没有相关内容"
- 返回回答文本 + 引用来源文件列表

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/taotao5433223-cell/rag-assistant.git
cd rag-assistant

# 安装依赖
pip install requests gradio
```

### 2. 配置 API 密钥

在 `src/` 目录下创建 `config.py`（已在 `.gitignore` 中，不会上传）：

```python
# 对话模型（智谱 GLM）
API_KEY = "你的智谱API密钥"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "GLM-5.3-Flash"

# 向量化模型（阿里云 DashScope）
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBED_API_KEY = "你的DashScope密钥"
EMBED_MODEL = "text-embedding-v4"
```

API 密钥获取：
- 智谱 GLM：[智谱开放平台](https://open.bigmodel.cn/) 注册后获取
- DashScope：[阿里云百炼](https://bailian.console.aliyun.com/) 开通后获取

### 3. 运行全流程

```bash
# 第一步：切分文档（约 1 秒）
python src/chunk.py

# 第二步：向量化（约 5-15 分钟，需联网调用 API）
python src/embed.py

# 第三步：启动问答界面
python src/app.py
```

浏览器打开 `http://127.0.0.1:7860`，输入问题即可对话。

### 4. 命令行模式（可选）

不启动网页界面，直接命令行问答：

```bash
python src/generate.py
```

## 使用示例

```
问：什么是注意力机制？
答：注意力机制是 Transformer 的核心组件，它允许模型衡量序列中不同
   单词之间的相对重要性……（结合检索到的笔记内容生成）

📚 来源：
📄 05-attention-basics.md
📄 06-causal-attention.md
📄 07-multi-head-attention.md
```

```
问：推荐一款适合玩游戏的笔记本电脑
答：资料中没有相关内容。

（知识库只覆盖大模型相关知识，不回答无关问题）
```

## 幻觉测试

幻觉测试是 RAG 系统的核心质量验证——确保模型不会对知识库外的内容"编造答案"。

### 测试方法

向系统提出 3 类**知识库中不存在**的问题，验证模型是否回复"资料中没有相关内容"而非胡编乱造。

### 测试用例与结果

| 编号 | 问题 | 问题类型 | 预期结果 | 实际结果 | 是否通过 |
|------|------|----------|----------|----------|----------|
| 1 | 推荐一款适合玩游戏的笔记本电脑 | 与知识库完全无关 | 拒绝回答 | 资料中没有相关内容 | ✅ 通过 |
| 2 | 北京有什么好吃的 | 与知识库完全无关 | 拒绝回答 | 资料中没有相关内容 | ✅ 通过 |
| 3 | 什么是 BERT 的 MLM 预训练任务 | 相关领域但笔记未覆盖（笔记只讲 GPT 路线） | 拒绝回答 | 资料中没有相关内容 | ✅ 通过 |

### 对照：知识库内的问题

| 问题 | 实际回答 | 来源 | 是否通过 |
|------|----------|------|----------|
| 什么是注意力机制？ | 结合笔记内容生成正确回答 | 05-attention-basics.md 等 | ✅ 通过 |
| BPE 分词是什么？ | 结合笔记内容生成正确回答 | 04-bpe-and-sliding-window.md | ✅ 通过 |
| GPT 的架构是什么？ | 结合笔记内容生成正确回答 | 02-transformer-and-gpt-architecture.md | ✅ 通过 |

### 结论

系统能正确区分"知识库内有/无"两种情况：
- **有资料**：结合检索内容生成准确回答 + 标注来源
- **无资料**：明确回复"资料中没有相关内容"，不编造

幻觉抑制的核心机制：
1. Prompt 约束："你只能根据下面的资料回答，资料里没有的内容，回答'资料中没有相关内容'"
2. `temperature=0`：确定性输出，降低随机编造概率
3. 检索结果作为唯一依据：LLM 只看到检索到的文档块，不使用自身预训练知识胡编

## 核心设计决策

### 为什么不用 LangChain？

使用框架可以一行搞定，但每一步的原理都被封装隐藏了。手写实现能深入理解切分策略为什么这样设计、余弦相似度怎么计算、Prompt 怎么拼接、检索结果怎么传给 LLM——每一行代码都是对 RAG 原理的落地。

### 为什么按标题切分而不是固定长度？

15 篇笔记有清晰的 `##`/`###` 标题层级，按标题切分能保证每个块是一个完整的知识单元（如"什么是注意力机制"一整节），比固定 800 字符切分语义更完整，检索精度更高。

### 为什么手写余弦相似度而不用 FAISS？

- 数据量小（296 个向量），纯 Python 遍历计算毫秒级完成，无需引入额外依赖
- 手写余弦公式和代码，深入理解向量相似度检索的数学原理
- 后续数据量增大时，可平滑替换为 FAISS/Milvus，接口不变

### 为什么用两个不同的 API？

- **对话用 GLM-5.3-Flash**：智谱免费额度充足，响应快，中文效果好
- **向量化用 text-embedding-v4**：阿里云 DashScope，1024 维，中文语义匹配准确
- 两者各有所长，组合使用效果最优；也可换成同一家服务

## 切分策略详解

`chunk.py` 的切分逻辑经过多轮优化，解决了以下问题：

| 问题 | 解决方案 |
|------|----------|
| 换行丢失 | `"\n".join(lines)` 保留原始换行 |
| 代码块被切断 | `in_code` 标志跟踪 ` ``` ` 状态，代码块内的 `###` 不当标题 |
| 代码围栏不平衡 | 切分时检查 ` ``` ` 奇偶数，奇数时移动切点到闭合围栏之后 |
| 空壳小节 | `len(content.strip()) < 30` 跳过只有标题无正文的空壳 |
| 切分残渣 | while 循环后再检查一次，过滤切完剩下的过短尾巴 |
| 标题提取误删 | `lstrip("# ")` 只剥左侧，避免标题末尾 `#` 被误删 |

## 文件说明

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/config.py` | 8 | API 密钥与模型配置 |
| `src/chunk.py` | 66 | 按标题切分文档，输出 `chunks.json` |
| `src/embed.py` | 37 | 调用 Embedding API 向量化，输出 `embeddings.json` |
| `src/vector_store.py` | 13 | 手写余弦相似度 + Top-K 检索 |
| `src/generate.py` | 38 | 问题向量化 → 检索 → 拼 Prompt → 调 LLM → 返回答案+来源 |
| `src/app.py` | 76 | Gradio 聊天界面，含自定义样式和示例问题 |

## 依赖

| 依赖 | 用途 | 安装 |
|------|------|------|
| `requests` | 调用 API | `pip install requests` |
| `gradio` | 网页界面 | `pip install gradio` |

无其他第三方依赖，Python 标准库（`json`/`os`/`math`）完成其余全部功能。

## 常见问题

**Q: 向量化太慢怎么办？**
A: 296 块约需 5-15 分钟（每块间隔 1 秒控频）。可先用前 3 篇笔记测试流程，通了再全量。

**Q: 检索结果不准怎么办？**
A: 调整 `top_k` 参数（`generate.py` 第 7 行），从 3 调到 5 或 8；检查切分是否合理。

**Q: 回答有幻觉怎么办？**
A: Prompt 已约束"只依据资料回答"。若仍有幻觉，降低 `temperature`（已设为 0），或收紧 Prompt 措辞。

**Q: 如何更换知识库？**
A: 把新的 Markdown 文件放入 `data/notes/`，重新运行 `chunk.py` → `embed.py` 即可。

## License

MIT
