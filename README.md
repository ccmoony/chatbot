# Chenbot
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
## 🚀 Fine-tuning
Navigate to the sft directory
```bash
cd sft
```
### Instruction-tuning
Run the following commands for instruction tuning using Alpaca-cleaned dataset:
```bash
python finetune.py --config config.yaml --use_lora False
```
Run the following commands for instruction tuning using Alpaca-cleaned dataset with LoRA:
```bash
python finetune.py --config config.yaml --use_lora True
```
### Style-tuning
Run the following commands to fine-tune the model with LoRA to imitate the tone of Zhen Huan:
```bash
python finetune.py --config config_style.yaml 
```
### Checkpoint
You can download our lora finetuned model from Here(https://jbox.sjtu.edu.cn/v/link/view/b4d703075cc944e1b2abae6d745a5e63).
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
python chat.py --model_path "your_model_path" --lora_path "zhenhuan_style_lora_path" --style
```
## 🎮 GUI Demo
Run the following command to start a Web UI demo:
```bash
python demo.py
```
The demo below will open in a browser on http://localhost:7860
