from langchain.tools import tool

@tool
def cal_kvcache_memory(batch_size: int, seq_len: int, num_layers: int, hidden_size: int) -> str:
    """
    用于计算大模型在长输入场景下的 KV Cache 显存占用量（单位：MB）。
    参数：
    - batch_size: 批次大小
    - seq_len: 序列长度（上下文长度）
    - hidden_size: 隐藏层维度
    - num_layers: 模型的层数
    """
    bytes_cost = batch_size * num_layers * 2 * seq_len * hidden_size * 2
    mb_cost = bytes_cost / (1024 * 1024)
    return f"当前配置下, KV Cache显存占用约为{mb_cost:.4f} MB"

tools = [cal_kvcache_memory]
