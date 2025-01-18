from chat import chat, load_model, stream_chat
import gradio as gr
from retrieval import retrieve_knowledge
device = "cuda:0"
title_html = '''
<img src="https://notes.sjtu.edu.cn/uploads/upload_84e031cf5eebf3c88f27c87c3d70788b.png" style="width: 150px; height: 150px;">
<h3>This is the chatbot of group ChatGPT
'''
model_path = "./models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
lora_path = "./Qwen2.5-3B-lora-output/20250101_204840_output/checkpoint-258800"
tokenizer, model = load_model(model_path, lora_path, device=device)
def load_models(model_type="chat"):
    global model, tokenizer, device
    model.cpu()
    model_path = "./models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    if model_type == "chat":
        system_prompt = "You are a useful AI assistant"
        lora_path = "./checkpoint/Qwen2.5-3B-lora-output/20250101_204840_output/checkpoint-258800"
    elif model_type == "huanhuan":
        system_prompt = "假设你是皇帝身边的女人--甄嬛。"
        lora_path = "./checkpoint/Qwen2.5-3B-lora-output/20250115_222004_output_style_finetune/checkpoint-18645"
    tokenizer, model = load_model(model_path, lora_path, device=device)
    return system_prompt

def chatbot_response(query, history, system_prompt, temperature, top_p, max_output_tokens, do_sample, use_retrieval, use_stream=True):
    if use_retrieval:
        knowledge = retrieve_knowledge(query)
    else:
        knowledge = ""
    if use_stream:
        for response, history in stream_chat(model, tokenizer, query, history, device=device, meta_instruction=system_prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_output_tokens, do_sample=do_sample, use_retrieval=use_retrieval):
            yield response, history, knowledge
    else:
        response, history = chat(model, tokenizer, query, history, device=device, meta_instruction=system_prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_output_tokens, do_sample=do_sample, use_retrieval=use_retrieval)
    
        return response, history, knowledge

def clear_history():
    return "", "", [], ""

chat_history = []

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(title_html)
            model_type = gr.Dropdown(["chat", "huanhuan"], label="Style", interactive=True)
            # gr.Markdown("## This is the chatbot of group ChatGPT")
            with gr.Accordion("Settings", open=False) as setting_row:
                system_prompt = gr.Textbox(
                    value="You are a useful AI assistant.",
                    label="System Prompt",
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_output_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=128,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )
                do_sample = gr.Checkbox(label="do_sample", value=True)
                use_retrieval = gr.Checkbox(label="use_retrieval", value=False)
            relevant_knowledge = gr.Textbox(label="Retrieved Knowledge", placeholder="set use_retrieval to true in settings if you want to use RAG..", lines=7)    

        with gr.Column(scale=8):
            chat_output = gr.Textbox(label="Chatbot", interactive=False, lines=20)
            user_input = gr.Textbox(label="User", placeholder="Enter message here...", lines=5)
            with gr.Row():
                submit_button = gr.Button("➡️ Send")
                clear_button = gr.Button("🗑️ Clear")
        

    submit_button.click(chatbot_response, inputs=[user_input, gr.State(chat_history), system_prompt, temperature, top_p, max_output_tokens, do_sample, use_retrieval], outputs=[chat_output, gr.State(chat_history), relevant_knowledge])
    clear_button.click(clear_history, inputs=[] ,outputs=[chat_output, user_input, gr.State(chat_history), relevant_knowledge])
    model_type.change(load_models, inputs=model_type, outputs=[system_prompt])
demo.launch()