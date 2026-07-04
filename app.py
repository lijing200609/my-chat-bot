from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json

app = Flask(__name__)

# 新版 OpenAI 客户端（支持图片多模态）
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

@app.route("/", methods=["GET"])
def home():
    return "Chat Bot is running! Send POST request to /chat"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json

        # 🔧 关键修改：兼容多种字段名
        user_text = data.get("text") or data.get("content") or data.get("message") or data.get("prompt") or ""
        user_image = data.get("image_base64", None)

        # 如果文字为空但有图片，自动补一个默认提示
        if not user_text and user_image:
            user_text = "请描述这张图片"

        # 如果文字仍然为空，返回错误提示
        if not user_text:
            return jsonify({"response": "请提供文字内容或图片", "status": "error"}), 400

        # 构建消息内容（文字 + 图片）
        content = [{"type": "text", "text": user_text}]
        if user_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_image}"}
            })

        # 调用 AIHubMix
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": content}],
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # 保存记忆
        memory = load_memory()
        memory.append({"role": "user", "content": user_text})
        memory.append({"role": "assistant", "content": reply})
        save_memory(memory)

        return jsonify({"response": reply, "status": "success"})

    except Exception as e:
        return jsonify({"response": f"错误：{str(e)}", "status": "error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
