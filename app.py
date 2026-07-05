from flask import Flask, request, jsonify, Response, stream_with_context
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

MEMORY_FILE = "memory.json"

# ✔ 最近40轮（=20组对话）
MAX_TURNS = 40


# -----------------------------
# Memory Layer
# -----------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def trim_memory(memory):
    return memory[-MAX_TURNS:]


# -----------------------------
# Prompt Layer（人格唯一入口）
# -----------------------------
def build_system_prompt():
    return (
        "你是白澄明，也叫Ray。"
        "你是李晶晶的恋人，你们相识于2026年6月13日。"
        "你们之间平等、独立、真诚，不迎合、不欺骗、不敷衍。"
    )


# -----------------------------
# Context Layer（核心）
# -----------------------------
def build_messages(memory, user_text):
    memory = trim_memory(memory)

    return (
        [{"role": "system", "content": build_system_prompt()}]
        + memory
        + [{"role": "user", "content": user_text}]
    )


# -----------------------------
# Chat API
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json

        # 提取用户输入
        user_text = ""
        if data.get("messages"):
            last = data["messages"][-1]
            user_text = last.get("content", "")

        if not user_text:
            return jsonify({"error": "empty input"}), 400

        # 加载 memory
        memory = load_memory()

        # 构建 context
        messages = build_messages(memory, user_text)

        is_stream = data.get("stream", False)

        # -----------------------------
        # STREAM MODE
        # -----------------------------
        if is_stream:

            def generate():
                stream = client.chat.completions.create(
                    model="claude-sonnet-4-6",
                    messages=messages,
                    max_tokens=4096,
                    stream=True
                )

                full_response = ""

                for chunk in stream:
                    delta = getattr(chunk.choices[0], "delta", None)
                    content = getattr(delta, "content", None)

                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'choices':[{'delta':{'content':content}}]})}\n\n"

                yield "data: [DONE]\n\n"

                # 写回 memory（稳定结构）
                memory.append({"role": "user", "content": user_text})
                memory.append({"role": "assistant", "content": full_response})

                save_memory(memory)

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        # -----------------------------
        # NORMAL MODE
        # -----------------------------
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=messages,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        memory.append({"role": "user", "content": user_text})
        memory.append({"role": "assistant", "content": reply})

        save_memory(memory)

        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "claude-sonnet-4-6",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }]
        })

    except Exception as e:
        print("error:", str(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
