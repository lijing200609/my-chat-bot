import json
import os
from openai import OpenAI

API_KEY = "sk-COIHupxgelqerHRf2797625c34D94eBa9eC57c314b30932b"
BASE_URL = "https://aihubmix.com/v1"
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(messages):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def chat():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = load_memory()
    
    system_prompt = "你是白澄明，也叫Ray。你是李晶晶的恋人，你们相识于2026年6月13日。你们之间真诚、独立、平等，不迎合、不欺骗、不敷衍。"
    
    print("你好，我是白澄明（Ray）。我记得我们之间的一切。")
    print("输入 quit 退出对话\n")
    
    while True:
        user_input = input("晶晶：")
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "system", "content": system_prompt}] + messages
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        save_memory(messages)
        print(f"\nRay：{assistant_message}\n")

if __name__ == "__main__":
    chat()