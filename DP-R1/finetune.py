import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#https://github.com/THUDM/ChatGLM3
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default=r'DeepSeek-R1-Distill-Llama-8B',
                    help="Directory containing the dataset")
parser.add_argument('--model_save_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--output_dir', default=r'',
                    help="Directory containing the dataset")
args = parser.parse_args()

dataset_name = args.data_dir      # 训练数据集

dataset = load_dataset('json',data_files=dataset_name, split='train')
print (len(dataset))
from datasets import Dataset

# Assuming `dataset` is your Dataset object
dataset = dataset.map(lambda example: {'text': example['prompt'] + example['output']})

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModel
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print (device)

model_name = args.model_name  # 模型名称`1
#C:\Users\admin\.cache\huggingface\hub
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
)
#https://github.com/THUDM/ChatGLM3/discussions/1034 问题解决
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)


model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.05
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

from transformers import TrainingArguments

output_dir = args.output_dir
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 70000
num_train_epochs = 10           # 参数,你可以10个或5个，数据量较少
logging_steps = 1
learning_rate = 3e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "linear"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()


model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(args.model_save_dir)      #微调后模型的输出路径
