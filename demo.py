from chat import chat, load_model, stream_chat
import gradio as gr

device = "cuda:0"
model_name = "/home/wanglonghao/wanglonghao_space/Projects/nlp_2024/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
lora_path = "/home/wanglonghao/wanglonghao_space/Projects/nlp_2024/Qwen2.5-3B-lora-output/20250101_204840_output/checkpoint-258800"
tokenizer, model = load_model(model_name, lora_path, device=device)
title_html = '''
<img src="https://notes.sjtu.edu.cn/uploads/upload_84e031cf5eebf3c88f27c87c3d70788b.png" style="width: 150px; height: 150px;">
<h3>This is the chatbot of group ChatGPT
'''
def chatbot_response(query, history, system_prompt, temperature, top_p, max_output_tokens, do_sample, use_stream=True):
    if use_stream:
        for response, history in stream_chat(model, tokenizer, query, history, device=device, meta_instruction=system_prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_output_tokens, do_sample=do_sample):
            yield response, history
    else:
        response, history = chat(model, tokenizer, query, history, device=device, meta_instruction=system_prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_output_tokens, do_sample=do_sample)
    
        return response, history

def clear_history():
    return "", "", []

chat_history = []

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(title_html)
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

        with gr.Column(scale=8):
            chat_output = gr.Textbox(label="Chatbot", interactive=False, lines=18)
            user_input = gr.Textbox(label="输入你的消息", placeholder="请在这里输入...", lines=5)
            with gr.Row():
                submit_button = gr.Button("发送")
                clear_button = gr.Button("🗑️ 清空")
        

    submit_button.click(chatbot_response, inputs=[user_input, gr.State(chat_history), system_prompt, temperature, top_p, max_output_tokens, do_sample], outputs=[chat_output, gr.State(chat_history)])
    clear_button.click(clear_history, inputs=[] ,outputs=[chat_output, user_input, gr.State(chat_history)])
demo.launch()