from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
import torch
import datetime

device = 'cuda' if cuda.is_available() else 'cpu'

model_name = 'model_path'
tokenizer = T5Tokenizer.from_pretrained('model_path')
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    input_ids = input_ids.to(device, dtype=torch.long)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)



filename = 'test_path'
fw_file = "output_path"
with open(filename,'r')as fr:
    with open(fw_file, 'w') as fw:
        for line in fr:
            answers = run_model(line.strip())
            fw.write(answers[0]+"\n")
