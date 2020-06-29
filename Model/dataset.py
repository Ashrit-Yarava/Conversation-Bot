#!/usr/bin/env python3
import torch
import json
import argparse
from random import shuffle, sample
from transformers import GPT2Tokenizer

from tqdm import tqdm

parser = argparse.ArgumentParser(description='make data.')
parser.add_argument('data_file', help='Path to dataset (JSON) file.')
parser.add_argument('size', type=int, help='Divde length by... 1 for all of data.')
parser.add_argument('save_file', help='Path to save data.')
args = parser.parse_args()

json_file = json.load(open(args.data_file))
shuffle(json_file)
length = len(json_file)
json_file = sample(json_file, length // args.size)

print("Json File now has {} conversations.".format(length // args.size))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


full_text = ''
for conversation in tqdm(json_file):
    for i in conversation:
        full_text += i['text'] + '<|endoftext|>'

print("Finished Collecting files.")
encoded_text = torch.LongTensor(tokenizer.encode(full_text))

torch.save(encoded_text, args.save_file)
