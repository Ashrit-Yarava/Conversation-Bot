import os
import json
file = json.load(open('datasets/TaskMaster.json'))

t = ''
for data in file:
    for text in data:
        t += text['speaker'] + ': ' + text['text'] + '\n'

open('datasets/For Tokenizer/TaskMaster.txt', 'w').write(t)

from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)
tokenizer.train(['datasets/For Tokenizer/wikitext.txt', 'datasets/For Tokenizer/TaskMaster.txt'], vocab_size=40000)

os.delete('datasets/For Tokenizer/TaskMaster.txt')