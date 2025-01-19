# CHENBOT
![demo](./logo.png)
## ⚙️ Configuration
First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/ccmoony/chatbot
cd chatbot
```
Then, create a new conda environment and install the dependencies:
```bash
conda create -n chenbot python=3.10
conda activate chenbot
pip install -r requirements.txt
```
Download [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) and put it in the root folder.
## 🚀 Fine-tuning
Navigate to the sft directory
```bash
cd sft
```
### Instruction-tuning
Run the following commands for instruction tuning using Alpaca-cleaned dataset:
```bash
python finetune.py
```
Run the following commands for instruction tuning using Alpaca-cleaned dataset with LoRA:
```bash
python finetune.py --use_lora
```
### Style-tuning
Run the following commands to fine-tune the model with LoRA to imitate the tone of Zhen Huan:
```bash
python finetune.py --config config_huanhuan.yaml --style huanhuan --use_lora
```
Run the following commands to fine-tune the model with LoRA to imitate the tone of Sun Wukong:
```bash
python finetune.py --config config_sunwukong.yaml --style wukong --use_lora
```
### Checkpoint
You can download our lora finetuned model from [Here](https://jbox.sjtu.edu.cn/v/link/view/b4d703075cc944e1b2abae6d745a5e63).
## 💻 Deploy on Terminal
Run the following commands to start the chatbot on terminal.
Use the pretrained model or the full-parameter fine-tuned model:
```bash
python chat.py --model_path "your_model_path"
```
Use the LoRA fine-tuned model:
```bash
python chat.py --model_path "your_model_path" --lora_path "your_lora_path"
```
Use the Zhen Huan styled model:
```bash
python chat.py --model_path "your_model_path" --lora_path "zhenhuan_style_lora_path" --style huanhuan
```
Use the Sun Wukong styled model:
```bash
python chat.py --model_path "your_model_path" --lora_path "wukong_style_lora_path" --style wukong
```
## 🎮 GUI Demo
Run the following command to start a Web UI demo(You may need to change the model path to your own path in demo.py):
```bash
python demo.py
```
The demo below will open in a browser on http://localhost:7860
