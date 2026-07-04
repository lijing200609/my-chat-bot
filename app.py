from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

def load_memory():
    """加载完整历史记忆（不限制条数）"""
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    """保存完整历史记忆"""
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

@app.route("/", methods=["GET"])
def home():
    return "Chat Bot is running! Send POST request to /chat"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        print(f"收到: {json.dumps(data, ensure_ascii=False)}")

        # 提取用户消息
        user_text = ""
        if data.get("messages") and len(data["messages"]) > 0:
            last_msg = data["messages"][-1]
            if last_msg.get("role") == "user":
                user_text = last_msg.get("content", "")

        if not user_text:
            return jsonify({"error": "请输入内容"}), 400

        # 1. 加载完整历史记忆（所有记录）
        history = load_memory()
        
        # 2. 构建完整消息列表：历史 + 当前用户消息
        messages = history + [{"role": "user", "content": user_text}]

        # 3. 调用 AI（把完整历史传过去）
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # 4. 保存记忆：追加新对话到 memory.json
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        save_memory(history)

        # 5. 返回标准 OpenAI 格式
        return jsonify({
            "choices": [
                {
                    "message": {
                        "content": reply
                    }
                }
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
