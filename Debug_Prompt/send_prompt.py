import requests
import argparse
import json

def send_prompt(prompt, url="http://localhost:8001/v1/chat/completions", model="Qwen2.5-14B-Instruct"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # 保证输出稳定
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        print("\n📜 Raw Response:\n")
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code} {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file")
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    send_prompt(prompt)
