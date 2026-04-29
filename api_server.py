import os
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer
from KvChat import LLM, SamplingParams
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

app = FastAPI(title="NanoKVLLM API Server")
class ChatMessage(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-1.7B"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 32000
    stream: Optional[bool] = False
    compress_enabled: Optional[bool] = True 

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llm = LLM(
    MODEL_PATH,
    enforce_eager=False,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages_dict,
        tokenize=False,
        enable_thinking=False,   # 思考模式
        add_generation_prompt=True 
    )

    llm.model_runner.config.kv_compress_enabled = request.compress_enabled
    sampling_params = SamplingParams(temperature=request.temperature, max_tokens=request.max_tokens)
    llm.add_request(prompt, sampling_params)

    if request.stream:
        async def generate_stream():
            token_buffer = []
            while not llm.is_finished():
                _, _, step_token_ids = llm.step()
                if step_token_ids is not None:
                    for tid in step_token_ids:
                        token_buffer.append(tid)
                        text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                        if '\ufffd' not in text:
                            chunk = {
                                "choices": [{"delta": {"content": text}}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            token_buffer.clear()
            if token_buffer:
                text = tokenizer.decode(token_buffer, skip_special_tokens=True)
                text = text.replace('\ufffd', '')
                if text:
                    chunk = {"choices": [{"delta": {"content": text}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    else:
        generated_token_ids = []
        while not llm.is_finished():
            _, _, step_token_ids = llm.step()
            if step_token_ids is not None:
                for tid in step_token_ids:
                    generated_token_ids.append(tid)
        assistant_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return {
            "choices": [{"message": {"role": "assistant", "content": assistant_text}}]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

