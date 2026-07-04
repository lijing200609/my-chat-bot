from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json

app = Flask(__name__)

# 从环境变量读取 API Key
client = OpenAI(
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

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
        
        # 构建消息内容
        content = [{"type": "text", "text": user_text}]
        if user_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_image}"}
            })
        
        # 调用 AIHubMix
        response = client.chat.completions.create(
            model="claude-sonnet-4.6",
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
