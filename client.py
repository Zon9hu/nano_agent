import requests
import json
import readline

def chat_client():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    # headers = {"Content-Type": "application/json"}
    # messages = [{"role": "system", "content": "你是一位资深的高级程序员，精通多种编程语言，可以根据用户的需求提供代码实现，稍后用户会将需求发给你。"}]
    messages = []
    print("========================================================")
    print("欢迎使用 NanoKVLLM API 客户端！(输入 'exit' 或 'quit' 退出)")
    print("========================================================")
    while True:
        try:
            user_input = input(">>>")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("期待下次再见，正在退出客户端...")
                break
            if not user_input.strip():
                continue
            messages.append({"role": "user", "content": user_input})
            data = {
                "model": "qwen3-1.7B",
                "messages": messages,
                "stream": False,
                "compress_enabled": True
            }
            print("Assistant: ", end="", flush=True)

            # response = requests.post(url, headers=headers, json=data)
            response = requests.post(url, json=data, stream=False)
            assistant_text = ""
            if data["stream"]:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:] # 去掉 "data: " 前缀
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                content_delta = data_json["choices"][0]["delta"].get("content", "")
                                assistant_text += content_delta
                                print(content_delta, end="", flush=True)
                            except json.JSONDecodeError:
                                pass    
            else:
                if response.status_code == 200:
                    data_json = response.json()
                    assistant_text = data_json["choices"][0]["message"]["content"]
                    print(assistant_text, end="", flush=True)
                else:
                    print(f"\n[请求失败] 状态码: {response.status_code}, 详情: {response.text}")
                    
            print() # 换行
            messages.append({"role": "assistant", "content": assistant_text})
            
        except KeyboardInterrupt:
            break
        except requests.exceptions.ConnectionError:
            print(f"无法连接到服务器,请检查API Server是否已启动")
        except Exception as e:
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    chat_client()
