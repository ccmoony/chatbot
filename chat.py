from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
from typing import List, Optional, Tuple, Union
from retrieval import retrieve_knowledge
try:
    from transformers.generation.streamers import BaseStreamer
except Exception:
    BaseStreamer = None
    
logging.set_verbosity_error()
def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    return tokenizer, model

def build_inputs(tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction="", retrieval=True):
    if history is None:
        history = []
    prompt = ""
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    if retrieval:
        knowledge = retrieve_knowledge(query)
        knowledge_context = "\n".join(knowledge)
        print(knowledge)
        prompt += f"<|im_start|>system\nRelevant knowledge: {knowledge_context}<|im_end|>\n"

    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True)

@torch.no_grad()
def chat(
        model,
        tokenizer,
        query: str,
        history: Optional[List[Tuple[str, str]]] = [],
        device: Optional[str] = "cuda:0",
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        meta_instruction: str = "You are a useful AI assistant",
        **kwargs,
    ):

        inputs = build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<|im_end|>")[0]
        history = history + [(query, response)]
        return response, history

def chatbot(model_name="/home/lihong/chenyuanjie/nlp2024/logs/20241214_122226_output/checkpoint-155280", device="cuda:0"):
    print("加载模型中，请稍候...")
    tokenizer, model = load_model(model_name, device)
    print("模型加载完成！输入 \\quit 结束会话，输入 \\newsession 开启新的对话。")
    
    history = []
    
    while True:

        query = input("用户: ")
        
        if query.strip().lower() == "\\quit":
            print("会话结束，再见！")
            break
        elif query.strip().lower() == "\\newsession":
            print("开启新的会话。")
            history = []
        else:
            try:
                response, history = chat(model, tokenizer, query, history, device)
                print("机器人: ", response)
            except Exception as e:
                print("生成回复时发生错误：", str(e))

chatbot(model_name="/home/lihong/chenyuanjie/nlp2024/logs/20241214_122226_output/checkpoint-155280")
