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
python finetune.py config.yaml
```

## 💻 Deploy on Terminal
Run the following command to start the chatbot on terminal:
```bash
python chat.py
```

## 🎮 Demo
Run the following command to start a WebUI demo:
```bash
python demo.py
```
