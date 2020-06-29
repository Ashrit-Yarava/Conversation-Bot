mkdir Model/Checkpoints/base
wget https://cdn.huggingface.co/gpt2-pytorch_model.bin
mv gpt2-pytorch_model.bin Model/Checkpoints/base/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json
mv gpt2-config.json Model/Checkpoints/base/config.json
mkdir Model/Checkpoints/current-model/
pip install -r requirements.txt
