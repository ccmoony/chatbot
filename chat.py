from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, TaskType, PeftModel,PeftConfig
from retrieval import retrieve_knowledge
try:
    from transformers.generation.streamers import BaseStreamer, TextStreamer
except Exception:
    BaseStreamer = None
import queue
import threading
import sys
import argparse
 
logging.set_verbosity_error()
def load_model(model_name, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval().to(device)
    if lora_path !=None:
        model = PeftModel.from_pretrained(model, lora_path)
    # config = PeftConfig.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    # model = PeftModel.from_pretrained(model, model_name).eval().to(device)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float32").eval().to(device)
    # lora_config = LoraConfig(
    #     r=8,                      # LoRA rank
    #     lora_alpha=32,           # LoRA alpha
    #     lora_dropout=0.05,       # LoRA dropout
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    #     target_modules = [
    #     "q_proj", "k_proj", "v_proj", "o_proj",
    #     ]
    # )
    # model = get_peft_model(model, lora_config)
    return tokenizer, model
    
def build_inputs(tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction="", retrieval=False):
    if history is None:
        history = []
    prompt = ""
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    if retrieval:
        knowledge = retrieve_knowledge(query)
        knowledge_context = "\n".join(knowledge)
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
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        meta_instruction: str = "You are a useful assistant.",
        device: Optional[str] = "cuda:1",
        use_retrieval: bool = False,
        **kwargs,
    ):

        inputs = build_inputs(tokenizer, query, history, meta_instruction, use_retrieval)
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

@torch.no_grad()
def stream_chat(
    model,
    tokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    meta_instruction: str = "You are a useful assistant.",
    device: Optional[str] = "cuda:1",
    use_retrieval: bool = False,
    **kwargs,
):
    if history is None:
        history = []
    """
    Return a generator in format: (response, history)
    Eg.
    ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
    ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
    """
    if BaseStreamer is None:
        raise ModuleNotFoundError(
            "The version of `transformers` is too low. Please make sure "
            "that you have installed `transformers>=4.28.0`."
        )

    response_queue = queue.Queue(maxsize=20)

    class ChatStreamer(BaseStreamer):
        """
        Streamer used in generate to print words one by one.
        """

        def __init__(self, tokenizer) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.queue = response_queue
            self.query = query
            self.history = history
            self.response = ""
            self.cache = []
            self.received_inputs = False
            self.queue.put((self.response, history + [(self.query, self.response)]))

        def put(self, value):
            if len(value.shape) > 1 and value.shape[0] > 1:
                raise ValueError("ChatStreamer only supports batch size 1")
            elif len(value.shape) > 1:
                value = value[0]

            if not self.received_inputs:
                # The first received value is input_ids, ignore here
                self.received_inputs = True
                return

            self.cache.extend(value.tolist())
            token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
            if token.strip() != "<|im_end|>":
                self.response = self.response + token
                print(token)
                history = self.history + [(self.query, self.response)]
                self.queue.put((self.response, history))
                self.cache = []
            else:
                self.end()

        def end(self):
            self.queue.put(None)

    def stream_producer():
        return chat(
            model=model,
            tokenizer=tokenizer,
            query=query,
            device=device,
            streamer=ChatStreamer(tokenizer=tokenizer),
            history=history,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            meta_instruction=meta_instruction,
            use_retrieval=use_retrieval,
            **kwargs,
        )

    def consumer():
        producer = threading.Thread(target=stream_producer)
        producer.start()
        while True:
            res = response_queue.get()
            if res is None:
                return
            yield res

    return consumer()

def chatbot(model_name, lora_path, device="cuda:0", system_prompt="You are a useful AI assistant", use_streamer = True, use_retrieval = False, style="chat"):
    if style=="huanhuan":
        system_prompt = "假设你是皇帝身边的女人--甄嬛。"
    elif style=="wukong":
        system_prompt = "假设你是西游记中的孙悟空。"
    print("加载模型中，请稍候...")
    tokenizer, model = load_model(model_name, lora_path, device)
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
                if use_streamer:
                    print("机器人: ")
                    sys.stdout.write("\033[s")  
                    sys.stdout.flush()
                    for response, history in stream_chat(model, tokenizer, query, history, device=device, use_retrieval=use_retrieval,meta_instruction=system_prompt):
                        sys.stdout.write(f"\033[u{response}")
                    print("\n")
                else:
                    response, history = chat(model, tokenizer, query, history, device=device, use_retrieval=use_retrieval, meta_instruction=system_prompt)  
                    print("机器人: \n", response)
            
            except Exception as e:
                print("生成回复时发生错误：", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chatbot deployed on terminal")
    parser.add_argument("--model_path", type=str, help="the path of pretrained model checkpoint")
    parser.add_argument("--lora_path", type=str, default=None, help="the path of lora checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="inference on which device")
    parser.add_argument("--style", type=str, default="chat", help="use style model")
    parser.add_argument("--retrieval", action="store_true", help="use RAG or not")
    args = parser.parse_args()
    chatbot(args.model_path, args.lora_path, args.device, use_retrieval=args.retrieval, style=args.style)




