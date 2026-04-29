from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model= "qwen3-1.7B",
    openai_api_base="http://127.0.0.1:8000/v1",
    openai_api_key="EMPTY",
    temperature=0.1
)

# 测试一下连通性
response = llm.invoke("用 C++ 写一个 Hello World")
print(response.content)
