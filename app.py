from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import time
import uuid

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

def load_memory():
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        app.logger.info(f"收到请求: {json.dumps(data, ensure_ascii=False)}")

        # 提取用户消息
        user_text = ""
        if data.get("messages") and len(data["messages"]) > 0:
            last_msg = data["messages"][-1]
            if last_msg.get("role") == "user":
                user_text = last_msg.get("content", "")

        if not user_text:
            return jsonify({"error": "请输入内容"}), 400

        # 加载完整历史
        history = load_memory()
        # 构建消息列表：历史 + 当前用户消息
        messages = history + [{"role": "user", "content": user_text}]

        # 调用 AIHubMix
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # 保存记忆（追加）
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        save_memory(history)

        # 构建标准 OpenAI 格式响应（包含所有必要字段，确保 ChatBox 能识别）
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "claude-sonnet-4-6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

    except Exception as e:
        app.logger.error(f"错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
