# from langchain_classic import hub
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage
from agent_tools import tools

llm = ChatOpenAI(
    model= "qwen3-1.7B",
    openai_api_base="http://127.0.0.1:8000/v1",
    openai_api_key="EMPTY",
    temperature=0.5,
)

# prompt = hub.pull("hwchase17/react")
agent = create_agent(llm, tools)
# agent = create_agent(
#     model=llm, 
#     tools=tools, 
#     system_prompt="你是一个有帮助的研究助理。"
#     )
    #"你是一个有帮助的研究助理。"
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("========================================")
    print("           ! Agent ! 启动 !")
    print("========================================")
    messages=[]
    query = "我正在对 Qwen 模型做长输入优化。请帮我计算一下，如果 batch_size 是 1，序列长度拓展到 128000，隐藏层维度是 4096，层数是 32，KV Cache 需要耗费多少 MB 的显存？"
    messages.append({"role": "user", "content": query})
    # print(messages)

    # result = agent.invoke({"messages": messages})
    # # print("\n最终结果: ", result["messages"][-1].content)
    # print(result)

    # for token, metadata in agent.stream(
    #         {"messages": messages},
    #         stream_mode="messages"
    #     ):
    #     if token.content:  # Check if there's actual content
    #         print(token.content, end="", flush=True)  # Print token

    # response = agent.invoke({
    #     "messages": [
    #         HumanMessage(content="你好，我是虎哥"),
    #         AIMessage(content="你好，虎哥，很高兴认识你。"),
    #         HumanMessage(content="我的名字是什么？")
    #     ]
    # })
    # print(response)

    # response = agent.invoke(
    #     {"messages": [HumanMessage(content=query)]},
    # )
    # for message in response['messages']:
    #     message.pretty_print()

    for token, metadata in agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="messages"
        ):
        # if token.content:  # Check if there's actual content
        print(token.content, end="", flush=True)  # Print token
