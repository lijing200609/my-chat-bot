# trigger redeploy
from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# 初始化 OpenAI 客户端（连接 AIHubMix）
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
        app.logger.info(f"收到完整请求: {json.dumps(data, ensure_ascii=False)}")

        # 兼容多种字段名提取用户文字
        user_text = (
            data.get("text") or 
            data.get("content") or 
            data.get("message") or 
            data.get("prompt") or 
            data.get("input") or 
            data.get("query") or 
            data.get("question") or
            data.get("messages")
        )

        # 如果 messages 是数组，尝试提取最后一条用户消息
        if not user_text and isinstance(data.get("messages"), list):
            for msg in reversed(data["messages"]):
                if msg.get("role") == "user":
                    user_text = msg.get("content")
                    break

        user_image = data.get("image_base64", None)

        # 如果文字为空但有图片，自动补一个默认提示
        if not user_text and user_image:
            user_text = "请描述这张图片"

        if not user_text:
            app.logger.warning("未提取到用户文字内容")
            return jsonify({"response": "请提供文字内容或图片", "status": "error"}), 400

        # 🔧 关键修改：content 构建为数组格式（符合 AIHubMix 要求）
        content_parts = [{"type": "text", "text": user_text}]
        if user_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_image}"}
            })

        messages = [{"role": "user", "content": content_parts}]

        # 调用 AIHubMix（模型名称必须与本地能跑通的一致）
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",   # 与你的本地 chat.py 保持一致
            messages=messages,
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
        app.logger.error(f"发生错误: {str(e)}")
        return jsonify({"response": f"错误：{str(e)}", "status": "error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
