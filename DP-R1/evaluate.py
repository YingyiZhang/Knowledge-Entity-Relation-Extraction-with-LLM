from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import argparse
print (device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--model_name', default=r'D:\hug\deepseek\DeepSeek-R1-Distill-Llama-8B',
                    help="Directory containing the dataset")
parser.add_argument('--model_save_dir', default=r'',
                    help="Directory containing the dataset")
parser.add_argument('--result_save_dir', default=r'',
                    help="Directory containing the dataset")
args = parser.parse_args()

from datasets import load_dataset

dataset_name = args.data_dir   #测试数据集

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True)  #原始模型路径
print ('aaa')
peft_model_id = args.model_save_dir   #参数路径
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_adapter()

model = model.to(device)

dataset = load_dataset('json',data_files=dataset_name, split='train')
dataset = dataset.map(lambda example: {'text': example['prompt'] + example['output']})

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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
        outputs = model.generate(**inputs, max_new_tokens=200)     #最大输出字符长度
        results = tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

        for result in results.split("\n"):
            if result.strip().startswith("### Response:"):
                result = result.split("### Response:")[1]
                print (str(i)+'+++++++++++results+++++++++++', result)
                fw.write(str(i)+'\t+++++++++++results+++++++++++:'+result.strip()+"\n")
                break
