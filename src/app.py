import gradio as gr
from generate import answer

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif !important;
}
#header {
    text-align: center;
    padding: 28px 0 20px;
    background: linear-gradient(135deg, #1a5276 0%, #2874a6 100%);
    border-radius: 12px;
    margin-bottom: 8px;
}
#header h1 {
    color: #fff !important;
    font-size: 1.8rem !important;
    margin-bottom: 4px !important;
}
#header p {
    color: #d6eaf8 !important;
    font-size: 0.95rem !important;
}
.footer {
    text-align: center;
    color: #aab7b8;
    font-size: 0.8rem;
    padding: 12px 0;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)


def chat(message, history):
    ans, srcs = answer(message)
    if "没有相关内容" in ans or "资料中" in ans and "没有" in ans:
        return ans
    source_md = "\n".join(f"📄 `{s}`" for s in srcs)
    reply = f"{ans}\n\n---\n**📚 来源：**\n{source_md}"
    return reply


with gr.Blocks(title="RAG 问答助手") as app:
    with gr.Column(elem_id="header"):
        gr.Markdown("# 大模型笔记知识库问答")
        gr.Markdown("《从零构建大模型》读书笔记 · RAG 检索增强生成")

    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=480, show_label=False),
        textbox=gr.Textbox(
            placeholder="输入你的问题，比如：什么是注意力机制？",
            container=False, scale=7,
        ),
        examples=[
            "什么是注意力机制？",
            "BPE 分词是什么？",
            "GPT 的架构是什么？",
            "预训练和微调有什么区别？",
            "什么是滑动窗口采样？",
        ],
    )

    gr.Markdown(
        '⚠️ 仅回答知识库内的内容，超出范围会回复"资料中没有相关内容"。',
        elem_classes="footer",
    )

if __name__ == "__main__":
    app.launch(theme=THEME, css=CSS)
