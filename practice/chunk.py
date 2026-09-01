import json
import os

DATA_DIR = "data/notes"

def split_sections(text):
    """
    按照'## ' 和'### '切分
    """
    sections = []
    cur_heading = "前言"
    cur_line = []
    in_Code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_Code = not in_Code
        if not in_Code and (line.startswith("## ") or line.startswith("### ")):
            if cur_line:
                sections.append((cur_heading, cur_line))
            cur_heading = line.lstrip("# ")
            cur_line = [line]
        else:
            cur_line.append(line)
    if cur_line:
        sections.append((cur_heading, cur_line))
    return sections

def make_chunks(text, max_length=1000):
    chunks = []
    for heading, lines in split_sections(text):
        content = "\n".join(lines)
        if len(content.strip()) < 30:
            continue
        while len(content) > max_length:
            cut = content.rfind("\n", 1000, max_length)
            if cut < 100:
                cut = max_length
            if content[:cut].count("```") % 2 == 1:
                close = content.find("```", cut)
                if close != -1:
                    cut = close + 3
            chunks.append(f"{heading}\n{content[:cut]}")
            content = content[cut:]
        if len(content.strip()) < 10:
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
        for chunk in make_chunks(text):
            all_chunks.append({"file": fn, "text": chunk})
    return all_chunks


if __name__ == "__main__":
    all_chunks = build_all_chunks()
    with open("practice/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"总块数：{len(all_chunks)}")
    print("前三块预览：\n")
    for c in all_chunks[:3]:
        print("---", c["file"], "|", c["text"].replace("\n", " "))