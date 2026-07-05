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
        print(f"收到请求: {json.dumps(data, ensure_ascii=False)}")

        # 提取用户消息
        user_text = ""
        if data.get("messages") and len(data["messages"]) > 0:
            last_msg = data["messages"][-1]
            if last_msg.get("role") == "user":
                user_text = last_msg.get("content", "")

        if not user_text:
            return jsonify({"error": "请输入内容"}), 400

        # 加载历史
        history = load_memory()
        messages = history + [{"role": "user", "content": user_text}]

        # 判断是否流式
        is_stream = data.get("stream", False)

        if is_stream:
            # 流式响应
            def generate():
                # 调用 AIHubMix 流式接口
                stream_response = client.chat.completions.create(
                    model="claude-sonnet-4-6",
                    messages=messages,
                    max_tokens=4096,
                    stream=True
                )
                # 收集完整回复用于保存记忆
                full_reply = ""
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_reply += content
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                yield "data: [DONE]\n\n"

                # 保存记忆
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": full_reply})
                save_memory(history)

            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        else:
            # 非流式响应
            response = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=messages,
                max_tokens=4096
            )
            reply = response.choices[0].message.content

            # 保存记忆
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            save_memory(history)

            resp = {
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
                ]
            }
            print(f"返回响应: {json.dumps(resp, ensure_ascii=False)}")
            return jsonify(resp)

    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
