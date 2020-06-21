import json
from glob import glob

files = glob('datasets/TaskMaster/*')
dump_file = 'datasets/TaskMaster.json'

conversations = []

for file in files:
    data = json.load(open(file))
    for conversation in data:
        temp = []
        for convo in conversation['utterances']:
            temp.append({'speaker': convo['speaker'], 'text': convo['text']})
        conversations.append(temp)

json.dump(conversations, open(dump_file, 'w'))