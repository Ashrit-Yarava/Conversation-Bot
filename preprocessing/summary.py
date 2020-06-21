import json

data = json.load(open('datasets/TaskMaster.json'))
num_conversations = len(data)
lengths = [len(i) for i in data]
conv_average = sum(lengths) / num_conversations

total = 0
totalw = 0
count = 0
for conv in data:
    for text in conv:
        count += 1
        total += len(text['text'])
        totalw += len(text['text'].split())
sent_average = round(total / count, 2)
word_average = round(totalw / count, 2)

print(f"Number of Conversations: {num_conversations}")
print(f"Average Length: {round(conv_average, 2)}")
print(f"Average length of each sentence: {sent_average}")
print(f"Average number of words: {word_average}")