import json
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

data = json.load(open('datasets/TaskMaster.json'))

full_text = ''
for conversation in data:
    for text in conversation:
        full_text += text['speaker'] + ': ' + text['text'] + '\n'
    full_text + '<|endoftext|>'

encoded_text = torch.LongTensor(tokenizer.encode(full_text))