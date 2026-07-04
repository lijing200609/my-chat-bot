from flask import Flask, request, jsonify
import openai
import os
import json

app = Flask(__name__)

# 设置 API 密钥和 Base URL（用环境变量）
openai.api_key = os.getenv("AIHUBMIX_API_KEY")
openai.api_base = "https://aihubmix.com/v1"   # 注意是 /v1 还是 /v1 取决于你的中转

# 加载记忆
def load_memory():
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 保存记忆
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
        user_text = data.get("text", "")
        user_image = data.get("image_base64", None)

        # 构建消息内容（旧版 API 不支持数组形式的 content，所以暂时不支持图片，
        # 如果你需要图片，稍后我们可以再调整，但先让文字跑起来）
        messages = [{"role": "user", "content": user_text}]

        # 调用 AIHubMix（使用旧版 completion 接口）
        response = openai.ChatCompletion.create(
            model="claude-sonnet-4.6",
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
        return jsonify({"response": f"错误：{str(e)}", "status": "error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
