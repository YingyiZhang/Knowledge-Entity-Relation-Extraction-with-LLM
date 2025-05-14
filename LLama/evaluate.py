from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print (device)
import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default=r'Llama-2-7b-chat-hf',
                    help="Directory containing the dataset")
parser.add_argument('--model_save_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--result_save_dir', default=r'',
                    help="Directory containing the dataset")
args = parser.parse_args()

dataset_name = args.data_dir    #测试数据集

from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(args.model_name,  quantization_config=bnb_config,
    trust_remote_code=True)  #原始模型路径
peft_model_id = args.model_save_dir   #参数路径
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_adapter()

model = model.to(device)

dataset = load_dataset('json',data_files=dataset_name, split='train')
dataset = dataset.map(lambda example: {'text': example['prompt'] + example['output']})

tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

with open(args.result_save_dir,"w",encoding='utf-8', errors='ignore') as fw:     #测试结果输出路径
    for i in range(0,len(dataset['output'])):
        output = dataset['output'][i]
        text = dataset['text'][i]
        prompt = dataset['prompt'][i]
        device = "cuda:0"
        print ('ori output',output)

        fw.write(str(i)+'\tori output:'+output+"\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100)     #最大输出字符长度

        results = tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

        result = results.split("### Response:")[1]
        print (str(i)+'+++++++++++results+++++++++++', result)
        fw.write(str(i)+'\t+++++++++++results+++++++++++:'+result.strip()+"\n")
