import json
import os

DATA_DIR = "data/notes"

def split_sections(text):
    """按 ## / ### 标题切成小节，返回 [(标题, 内容)]"""
    sections = []
    cur_heading = "前言"
    cur_lines = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and (line.startswith("## ") or line.startswith("### ")):
            if cur_lines:
                sections.append((cur_heading, cur_lines))
            cur_heading = line.lstrip("# ").strip()
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append((cur_heading, cur_lines))
    return sections

def make_chunks(text, max_len=800):
    """每个小节一个块，超长的块按照换行再切一刀"""
    chunks = []
    for heading, lines in split_sections(text):
        content = "\n".join(lines)
        if len(content.strip()) < 30:
            continue
        while len(content) > max_len:
            cut = content.rfind("\n", 100, max_len)
            if cut < 100:
                cut = max_len
            if content[:cut].count("```") % 2 == 1:
                close = content.find("```", cut)
                if close != -1:
                    cut = close + 3
            chunks.append(f"{heading}\n{content[:cut]}")
            content = content[cut:]
        if len(content.strip()) < 30:
            continue
        chunks.append(f"{heading}\n{content}")
    return chunks

def build_all_chunks():
    all_chunks = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(DATA_DIR, fn)) as f:
            text = f.read()
        for c in make_chunks(text):
            all_chunks.append({"file": fn, "text": c})
    return all_chunks

if __name__ == "__main__":
    chunks = build_all_chunks()
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"总块数：{len(chunks)}")
    print("前三块预览：\n")
    for c in chunks[:3]:
        print("---", c["file"], "|", c["text"][:40].replace("\n", " "))