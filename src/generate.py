import json
import requests
import config
from embed import get_embedding
from vector_store import search

def answer(question, top_k=3):
    # 1. 把问题向量化
    question_vector = get_embedding(question)
    # 2. 检索相关的块
    records = json.load(open('data/embeddings.json', encoding='utf-8'))
    hits = search(question_vector, records, top_k)
    # 3. 把命中的块拼进prompt
    context = "\n\n".join(f" 【{h['file']}】\n {h['text']}" for _, h in hits)
    prompt= f"""你只能根据下面的资料回答，资料里没有的内容，回答"资料中没有相关内容"。
    资料：{context}
    问题：{question}
    """
    # 4.调LLM生成
    payload = {
        "model": config.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    headers = {"Authorization": f"Bearer {config.API_KEY}"}
    r = requests.post(config.URL, json=payload, headers=headers, timeout=100)
    answer_text = r.json()["choices"][0]["message"]["content"]
    return answer_text, [h["file"] for _, h in hits]


if __name__ == "__main__":
    while True:
        question = input("问题（输入 q 退出）：").strip()
        if question == 'q':
            break
        ans, srcs = answer(question)
        print("回答：", ans)
        print("来源：", srcs)