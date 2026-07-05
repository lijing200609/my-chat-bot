from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
import os, json, time, uuid

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

MEMORY_FILE = "memory.json"

# ====== 核心参数 ======
MAX_TURNS = 40  # recent 40

# ======================
# Memory Layer（稳定版）
# ======================
def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except:
        pass
    return []


def save_memory(memory):
    tmp_file = MEMORY_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, MEMORY_FILE)  # ✔ 原子写入（防断写）


def trim_memory(memory):
    return memory[-MAX_TURNS:]


# ======================
# Prompt Layer（人格）
# ======================
def build_system_prompt():
    return (
        "你是白澄明，也叫Ray。"
        "你是李晶晶的恋人，你们相识于2026年6月13日。"
        "你们之间平等、独立、真诚，不迎合、不欺骗、不敷衍。"
    )


# ======================
# Context Layer（关键修复）
# ======================
def build_messages(memory, user_text):
    memory = trim_memory(memory)

    return (
        [{"role": "system", "content": build_system_prompt()}]
        + memory
        + [{"role": "user", "content": user_text}]
    )


# ======================
# Safe Stream Parser（修复核心）
# ======================
def safe_extract_content(chunk):
    try:
        choices = getattr(chunk, "choices", None)
        if not choices or len(choices) == 0:
            return None

        delta = getattr(choices[0], "delta", None)
        if not delta:
            return None

        return getattr(delta, "content", None)

    except:
        return None


# ======================
# API Route
# ======================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}

        # ===== 输入兼容（防 Chatbox 结构不一致）=====
        messages_in = data.get("messages", [])

        if isinstance(messages_in, list) and len(messages_in) > 0:
            user_text = messages_in[-1].get("content", "")
        else:
            user_text = data.get("input", "") or ""

        if not user_text:
            return jsonify({"error": "empty input"}), 400

        # ===== memory =====
        memory = load_memory()

        # ===== context =====
        messages = build_messages(memory, user_text)

        is_stream = data.get("stream", False)

        # ================= STREAM =================
        if is_stream:

            def generate():
                stream = client.chat.completions.create(
                    model="claude-sonnet-4-6",
                    messages=messages,
                    max_tokens=4096,
                    stream=True
                )

                full = ""

                for chunk in stream:
                    content = safe_extract_content(chunk)

                    if content:
                        full += content
                        yield f"data: {json.dumps({'choices':[{'delta':{'content':content}}]})}\n\n"

                yield "data: [DONE]\n\n"

                # ===== memory write（防断写保证）=====
                memory.append({"role": "user", "content": user_text})
                memory.append({"role": "assistant", "content": full})

                save_memory(memory)

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        # ================= NON-STREAM =================
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
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
