# Conversation Bot
---
Better Chatbots using AI.
**NOTE: The model is currently only trained on the TaskMaster dataset.**

## Training
---
The model can be trained using the Model/train.py script.
The preprocessing/ directory contains scripts to convert unclean data to clean data.
Save the encoded data to a .pt file.

## Inference
---
`conversation.py` provides a simple inference to check if the model works as expected.
Conversations can be saved to the Logs directory.

## Add files
---
Create a directory inside the datasets and preprocessing for converting and analyzing data.

## Training on Colaboratory
---
To train the model on google colab, first run the `bash colab-setup.sh` to set up the environment.

