import pandas as pd
import json
import os
import time
import requests

# 配置路径
content_path = "cora.content"
node_info_path = "node_info.csv"
output_path = "cora_text_graph_simplified.jsonl"

# vLLM 推理地址（你可改成 OpenAI 或其他）
VLLM_API_URL = "http://localhost:8001/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instruct"

# 简化提示模版
SYSTEM_PROMPT = "You are a helpful assistant that condenses academic paper descriptions."
USER_PROMPT_TEMPLATE = (
    "Given the following title and abstract, create a concise ID sentence that represents the paper's topic. "
    "Avoid redundancy and keep it short and informative.\n\n"
    "{text}\n\nConcise ID:"
)

def call_vllm(text):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ],
        "temperature": 0.2,
        "max_tokens": 128
    }
    try:
        resp = requests.post(VLLM_API_URL, json=payload, timeout=20)
        resp.raise_for_status()
        response = resp.json()
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"⚠️ Failed to simplify text: {e}")
        return ""

# 加载原始数据
content_df = pd.read_csv(content_path, sep="\t", header=None)
content_df.columns = ["paper_id"] + [f"feat_{i}" for i in range(content_df.shape[1] - 2)] + ["label"]

info_df = pd.read_csv(node_info_path)
pid_to_label = dict(zip(content_df["paper_id"], content_df["label"]))
pid_to_text = {}
for _, row in info_df.iterrows():
    pid = row["paper_id"]
    title = str(row["title"]).strip().replace("\n", " ")
    abstract = str(row["abstract"]).strip().replace("\n", " ")
    pid_to_text[pid] = f"Title: {title}\nAbstract: {abstract}"

# 构建 jsonl 并进行文本简化
with open(output_path, "w") as f:
    for node_id, pid in enumerate(content_df["paper_id"]):
        if pid not in pid_to_text:
            continue
        raw_text = pid_to_text[pid]
        simplified = call_vllm(raw_text)
        record = {
            "node_id": node_id,
            "paper_id": int(pid),
            "text": simplified,
            "label": pid_to_label[pid]
        }
        f.write(json.dumps(record) + "\n")
        print(f"✅ Node {node_id} simplified: {simplified[:60]}...")
        time.sleep(0.2)  # 控制请求频率

print(f"🎉 All nodes saved to {output_path}")
