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
        user_text = None
        
        # 优先从 messages 里提取
        if isinstance(data.get("messages"), list):
            for msg in reversed(data["messages"]):
                if msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        user_text = content
                    elif isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                user_text = part.get("text")
                                break
                    break
        
        if not user_text:
            user_text = (
                data.get("text") or 
                data.get("content") or 
                data.get("message") or 
                data.get("prompt") or 
                data.get("input") or 
                data.get("query") or 
                data.get("question") or
                ""
            )

        user_image = data.get("image_base64", None)

        if not user_text and user_image:
            user_text = "请描述这张图片"

        if not user_text:
            app.logger.warning("未提取到用户文字内容")
            return jsonify({"response": "请提供文字内容或图片", "status": "error"}), 400

        content_parts = [{"type": "text", "text": user_text}]
        if user_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_image}"}
            })

        messages = [{"role": "user", "content": content_parts}]

        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        memory = load_memory()
        memory.append({"role": "user", "content": user_text})
        memory.append({"role": "assistant", "content": reply})
        save_memory(memory)

        # ✅ 关键修改：返回标准 OpenAI 格式
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
        app.logger.error(f"发生错误: {str(e)}")
        return jsonify({"response": f"错误：{str(e)}", "status": "error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
