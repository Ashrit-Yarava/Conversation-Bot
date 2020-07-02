import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored, cprint

def get_input(): # Simple user input.
    inpt = input(colored('>>> ', "yellow"))
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


def main():
    
    parser = argparse.ArgumentParser(description='Talk to your trained model.')
    parser.add_argument('model_dir', type=str, help='Path to a transfomers model checkpoint. Type help for more info.')
    args = parser.parse_args()

    if args.model_dir == 'help':
        help_text = """
    A model checkpoint is a directory containing a pytorch_model.bin file and a config.json file.
    This directory will be used to load the model. Make sure that the model is an autoregressive model.
        """
        print(help_text)
        exit(0)

    cprint("Conversation Bot", "cyan")
    cprint("================================", "cyan")
    
    gpt2, tokenizer = AutoModelForCausalLM.from_pretrained(args.model_dir), AutoModelForCausalLM.from_pretrained('distilgpt2')
    inpt = get_input()
    previous_inpts = [inpt]

    while inpt != -1:
        output = get_bot_output(gpt2, tokenizer, previous_inpts)
        previous_inpts.append(output)
        print(colored('Bot: ', 'green') + output)
        inpt = get_input()
        previous_inpts.append(inpt)

if __name__ == '__main__':
    main() # Run the actual program.
