import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
dataset = json.load(open('datasets/TaskMaster.json'))

length = len(dataset)
dataset = dataset[:len(dataset) // 5]

text_data = ''
for conversation in tqdm(dataset):
    for text in conversation:
        text_data += text['speaker'].lower() + ": " + text['text'] + '\n'
    text_data += '<|endoftext|>\n'

encoded_text = torch.LongTensor(tokenizer.encode(text_data))
torch.save(encoded_text, 'datasets/TaskMaster.pt')