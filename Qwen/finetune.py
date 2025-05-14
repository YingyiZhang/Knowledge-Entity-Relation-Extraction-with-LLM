'''
使用transformer的trainer和lora进行微调
'''
import torch
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 配置路径（根据实际路径修改）
model_path = r"Qwen"
# 模型路径
data_path_train = r"train_path"
data_path_test = r"test_path"
# 数据集路径
output_path = r""
# 微调后模型保存路径
# 强制使用GPU
assert torch.cuda.is_available(), "必须使用GPU进行训练！"
device = torch.device("cuda")


# 自定义回调记录Loss
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


# 数据预处理函数
def process_data(tokenizer):
    dataset = load_dataset("json", data_files=data_path_train, split='train')

    def format_example(example):
        instruction = f"{example['prompt']} "
        inputs = tokenizer(
            f"{instruction}{example['output']}",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        ins_encoded = tokenizer(
            instruction,
            add_special_tokens = False
        )["input_ids"]
        ins_length = len(ins_encoded)

        labels = inputs["input_ids"].clone().squeeze(0)
        labels[:ins_length] = -100
        print (labels)

        return {"input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels
                }

    return dataset.map(format_example, remove_columns=dataset.column_names)

# LoRA配置
peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        #target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
)

# 训练参数配置
training_args = TrainingArguments(
        output_dir=output_path,
        save_steps=5000,  #每几步存一次模型#14157
        per_device_train_batch_size=1,  # 显存优化设置
        gradient_accumulation_steps=4,  # 累计梯度相当于batch_size=8
        num_train_epochs=10,
        learning_rate=3e-4,
        logging_steps=20,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="linear",
        no_cuda=False,  # 强制使用CUDA
        dataloader_pin_memory=False,  # 加速数据加载
)

def main():
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        # 加载模型到GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device}  # 强制使用指定GPU
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        # 准备数据
        dataset = process_data(tokenizer)
        # 训练回调
        loss_callback = LossCallback()

        # 数据加载器
        def data_collator(data):
            batch = {
                "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
                "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
                "labels": torch.stack([torch.tensor(d["labels"]) for d in data]).to(device)
                # 使用input_ids作为labels
            }

            return batch
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[loss_callback]
        )
        # 开始训练
        print("开始训练...")
        trainer.train()
        # 保存最终模型
        trainer.model.save_pretrained(output_path)
        print(f"模型已保存至：{output_path}")
        # 绘制训练集损失Loss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(loss_callback.losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(output_path, "loss_curve.png"))
        print("Loss曲线已保存")

if __name__ == "__main__":
        main()