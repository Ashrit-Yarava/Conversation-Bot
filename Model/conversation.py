import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description='A simple application for testing bot.')
parser.add_argument('model_dir', type=str, help='Path to model.')
parser.add_argument('log_file', type=str, help='Path to log file.')

"""
model_dir ==> directory for the pretrained mdoel.
log_file ==> Path to write the conversation to.
"""

args = parser.parse_args()

gpt2 = AutoModelForCausalLM.from_pretrained(args.model_dir) # Load Model.
tokenizer = AutoTokenizer.from_pretrained('distilgpt2') # Load pretrained tokenizer.

def get_input(): # Simple user input.
    inpt = input('>>> ')
    if inpt == 'exit':
        return -1
    return inpt

def process_text(inpt, tokenizer): # Convert previous inpt to text and encode it.
    full_text = ''
    for i in inpt:
        full_text += i + '<|endoftext|>'
    encoded_text = tokenizer.encode(full_text)
    return encoded_text

def convert(text): # convert list to torch tensor.
    return torch.LongTensor([text])

def single_step(model, encoded_text): # Get an output token.
    return torch.argmax(model(encoded_text)[0][:, -1, :]).detach().tolist()

def get_bot_output(model, tokenizer, previous_input):
    starting_input = process_text(previous_input, tokenizer)
    token = single_step(model, convert(starting_input))

    model_output = []
    
    while token != 50256:
        print(token)
        starting_input.append(token)
        token = single_step(model, convert(starting_input))
        model_output.append(token)
        starting_input.append(token)

    return tokenizer.decode(model_output[:-1])


print("Conversation Bot")
print("================================")

inpt = get_input()
previous_inpts = [inpt]
while inpt != -1:
    output = get_bot_output(gpt2, tokenizer, previous_inpts)
    previous_inpts.append(output)
    print('-'*10)
    print(output)
    print('-'*10)
    inpt = get_input()
    previous_inpts.append(inpt)

print("Thanks for talking!")
full_text = '\n'.join(previous_inpts[:-1])
open(args.log_file, 'w').write(full_text)
