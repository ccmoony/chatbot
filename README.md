# Chatbot
cs.jsonl can be downloaded [[here]](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main)
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
Run the following commands to fine-tune the model:
```bash
cd sft
python finetune.py --config config.yaml --use_lora False
```

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

## 🎮 GUI Demo
Run the following command to start a Web UI demo:
```bash
python demo.py
```
The demo below will open in a browser on http://localhost:7860
