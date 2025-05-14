import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json

# 设置可见GPU设备（根据实际GPU情况调整）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定仅使用GPU

# 路径配置 ------------------------------------------------------------------------
base_model_path = r"Qwen"  # 原始预训练模型路径
peft_model_path = r""  # LoRA微调后保存的适配器路径

# 模型加载 ------------------------------------------------------------------------
# 初始化分词器（使用与训练时相同的tokenizer）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载基础模型（半精度加载节省显存）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # 使用float16精度
    device_map="auto"  # 自动分配设备（CPU/GPU）
)

# 加载LoRA适配器（在基础模型上加载微调参数）
lora_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
# 合并LoRA权重到基础模型（提升推理速度，但会失去再次训练的能力）
lora_model = lora_model.merge_and_unload()
#lora_model.eval()  # 设置为评估模式


# 生成函数 ------------------------------------------------------------------------
def generate_response(model, prompt):
    """统一的生成函数
    参数：
        model : 要使用的模型实例
        prompt : 符合格式要求的输入文本
    返回：
        清洗后的回答文本
    """
    # 输入编码（保持与训练时相同的处理方式）
    #print (prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",  # 返回PyTorch张量
        max_length=512,  # 最大输入长度（与训练时一致）
        truncation=True,  # 启用截断
        padding="max_length"  # 填充到最大长度（保证batch一致性）
    ).to(model.device)  # 确保输入与模型在同一设备

    # 文本生成（关闭梯度计算以节省内存）
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,  # 生成内容的最大token数（控制回答长度）
            temperature=0.1,  # 温度参数（0.0-1.0，值越大随机性越强）
            top_p=0.9,  # 核采样参数（保留累积概率前90%的token）
            repetition_penalty=1.1,  # 重复惩罚系数（>1.0时抑制重复内容）
            eos_token_id=tokenizer.eos_token_id,  # 结束符ID
            pad_token_id=tokenizer.pad_token_id,  # 填充符ID
        )

    # 解码与清洗输出
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 跳过特殊token
    print (full_text)
    answer = full_text.split("### answer：\n")[-1].strip().split("<|endoftext|")[0]  # 提取答案部分
    return answer


# 对比测试函数 --------------------------------------------------------------------
def compare_models(question):
    """模型对比函数
    参数：
        question : 自然语言形式的医疗问题
    """
    # 构建符合训练格式的prompt（注意与训练时格式完全一致）
    prompt = f"{question}"

    # 双模型生成
    lora_answer = generate_response(lora_model, prompt)  # 微调模型

    print("\n" + "=" * 50)  # 分隔线
    print(f"\033[1;32m[LoRA模型]\033[0m\n{lora_answer}")  # 绿色显示微调模型结果
    print("=" * 50 + "\n")
    return lora_answer


# 主程序 ------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试问题集（可自由扩展）
    finetune_output = "output_path"

    data_path_test = r"test_path"
    test_questions = []
    with open(finetune_output,'w', encoding='utf-8', errors='ignore') as finetune_fw:
        with open(data_path_test,"r", encoding='utf-8', errors='ignore') as fr:
            for i, line in enumerate(fr):
                line_dict = json.loads(line)
                instruction = f"{line_dict['prompt']} "
                test_questions.append(instruction)
        # 遍历测试问题
        for i, q in enumerate(test_questions):
            lora_answer = compare_models(q)
            finetune_fw.write(str(i)+"\t"+lora_answer + "\n")
