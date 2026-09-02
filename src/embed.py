import json
import requests
import time
import config

def get_embedding(text, max_retry=3):
    payload = {
        "model": config.EMBED_MODEL,
        "input": text
    }
    headers = {
        "Authorization": f"Bearer {config.EMBED_API_KEY}"
    }
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.post(config.EMBED_URL, json=payload, headers=headers, timeout=60)
            return r.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"第{attempt}次尝试：{e}")
            time.sleep(2 * attempt)
    return []

def embed_all():
    chunks = json.load(open("data/chunks.json", encoding="utf-8"))
    records = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk["text"])
        records.append({**chunk, "vector": vec})
        print(f"{i+1}/{len(chunks)}已完成",flush=True)
        time.sleep(1)
    with open("data/embeddings.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"向量化完成，共{len(records)} 块")


if __name__ == "__main__":
    embed_all()