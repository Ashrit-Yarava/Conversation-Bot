import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description='A simple application for testing bot.')
parser.add_argument('model_dir', type=str, help='Path to model.')

args = parser.parse_args()

gpt2 = AutoModelForCausalLM.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

def get_input():
    inpt = input('>>> ')
    if inpt == 'exit':
        return -1
    return inpt

def process_text(inpt, tokenizer):
    full_text = ''
    for i in inpt:
        full_text += i + '<|endoftext|>'
    encoded_text = tokenizer.encode(full_text)
    return encoded_text

def convert(text):
    return torch.LongTensor([text])

def single_step(model, encoded_text):
    return torch.argmax(model(encoded_text)[0][:, -1, :]).detach().tolist()

def get_bot_output(model, tokenizer, previous_input):
    starting_input = process_text(previous_input, tokenizer)
    token = single_step(model, convert(starting_input))

    model_output = []
    
    while token is not 50256:
        print(token)
        starting_input.append(token)
        token = single_step(model, convert(starting_input))
        model_output.append(token)
        starting_input.append(token)

    return tokenizer.decode(model_output)


# print("Conversation Bot")
# print("================================")

# inpt = get_input()
# while inpt != -1:

