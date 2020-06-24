import sys
import argparse
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer


def generate(model, tokenizer, previous_text, user_input):
    print(previous_text.shape)
    output = model.generate(previous_text,
                            # num_beams=5,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=50256,
                            # temperature=0.9,
                            # no_repeat_ngram_size=3
                            )
    output = tokenizer.decode(output[0].tolist())
    print(output)
    output = output.split('\n')
    for i in range(len(output)):
        if user_input in output[i]:
            output = output[i + 1]
    # print(output)
    while 'assistant: ' in output:
        output = output.replace('assistant: ', '')
    return output


def get_input():
    return input(">>> ")


def main(args):
    model = AutoModelWithLMHead.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

    chatbot_output = ''
    full_text = ''
    while '<|endoftext|>' not in chatbot_output:
        user_input = get_input()
        if user_input == 'exit':
            sys.exit()
        full_text += 'user: ' + user_input + '\n' + 'assistant: '
        encoded_text = torch.LongTensor([tokenizer.encode(full_text)])
        generated_output = generate(model, tokenizer, encoded_text, user_input)
        print('-' * 10)
        print(generated_output)
        print('-' * 10)
        chatbot_output += generated_output
        full_text += generated_output + '\n'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple Application to communicate with chatbot.')
    parser.add_argument('model_path', type=str,
                        help='Path to the directory containing the model dump.')
    arguments = parser.parse_args()
    main(arguments)
