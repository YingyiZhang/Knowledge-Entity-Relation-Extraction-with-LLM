import datetime
import json
import os
import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AdamW
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import classification_report, f1_score
import transformers


IRRELEVANT_LABEL = "(J) no-relation"
T_DIVISOR = 3


def build_deepseek_prompt(input_text, target_label=None):


    instruction = "Classify the relation of the following text:"

    input_part = f"<｜begin of sentence｜>{instruction} {preprocess(input_text)}<｜end of sentence｜>"

    if target_label is not None:

        return f"{input_part}{preprocess(target_label)}<｜end of sentence｜>"
    else:

        return input_part


def build_prompt(input_text, target_label=None):

    return build_deepseek_prompt(input_text, target_label)

def readFile(sen_file, label_file):

    sens = []
    labs = []
    with open(sen_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            sens.append(line.strip())
    with open(label_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            labs.append(line.strip())

    original_pairs = [(s, l) for s, l in zip(sens, labs)]
    relevant_pairs_by_label = {}
    irrelevant_pairs = []

    for pair in original_pairs:
        label = pair[1].strip().lower()
        if label == IRRELEVANT_LABEL.lower():
            irrelevant_pairs.append(pair)
        else:
            if label not in relevant_pairs_by_label:
                relevant_pairs_by_label[label] = []
            relevant_pairs_by_label[label].append(pair)

    N = len(irrelevant_pairs)
    print(f"Original counts - Irrelevant: {N}")

    balanced_pairs = []
    for label, pairs_for_this_label in relevant_pairs_by_label.items():
        M_i = len(pairs_for_this_label)
        replication_times = max(0, int(N / (T_DIVISOR * M_i)))
        print(f"Label '{label}': Replication times: {replication_times}")

        for pair in pairs_for_this_label:
            for _ in range(replication_times):
                balanced_pairs.append(pair)

    balanced_pairs.extend(irrelevant_pairs)
    print(f"Final dataset size: {len(balanced_pairs)}")
    return balanced_pairs


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")


def evaluate(model, tokenizer, data_path, device, desc="Validation", batch_size=1, output_file=None):
    print(f"Starting {desc}...")
    sen_file = os.path.join(data_path, 'sentence')
    label_file = os.path.join(data_path, 'label')

    eval_data = []
    with open(sen_file, 'r', encoding='utf-8') as fr_sen, open(label_file, 'r', encoding='utf-8') as fr_lab:
        for sen_line, lab_line in zip(fr_sen, fr_lab):
            eval_data.append((sen_line.strip(), lab_line.strip()))

    model.eval()
    predictions = []
    true_labels = []
    results_to_save = []
    total_loss = 0
    with torch.no_grad():
        for step in range(0, len(eval_data), batch_size):
            batch = eval_data[step:step + batch_size]
            input_texts = []
            batch_true_labels = []

            for input_text, target_label in batch:
                clean_input = preprocess(input_text)
                clean_target = preprocess(target_label)
                full_prompt = build_prompt(clean_input, None)
                input_texts.append(full_prompt)
                batch_true_labels.append(clean_target.lower())

            encodings = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            generated_ids = model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_new_tokens=64,
                num_beams=2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i, (full_output, (orig_input, _)) in enumerate(zip(batch_preds, batch)):
                prompt_text = tokenizer.decode(encodings.input_ids[i], skip_special_tokens=True)

                if full_output.startswith(prompt_text):
                    pred_label = full_output[len(prompt_text):].strip().lower()
                else:
                    pred_label = full_output.split("Output:")[
                        -1].strip().lower() if "Output:" in full_output else full_output.strip().lower()

                predictions.append(pred_label)
                true_labels.append(batch_true_labels[i])

                results_to_save.append({
                    "input": orig_input,
                    "true_label": batch_true_labels[i],
                    "prediction": pred_label
                })

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results_to_save:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Results saved to {output_file}")


    exact_matches = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    accuracy = exact_matches / len(eval_data)


    unique_labels = sorted(list(set(true_labels + predictions)))
    macro_f1 = f1_score(true_labels, predictions, labels=unique_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, labels=unique_labels, average='micro', zero_division=0)


    report = classification_report(
        true_labels,
        predictions,
        labels=unique_labels,
        target_names=unique_labels,
        zero_division=0,
        output_dict=True
    )

    print(f"\n{desc} Results:")
    print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

    print("\n--- Detailed Classification Report ---")
    print(f"{'Label':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    for label in unique_labels:
        if label in report:
            data = report[label]
            print(
                f"{label:<15} {data['precision']:<10.4f} {data['recall']:<10.4f} {data['f1-score']:<10.4f} {data['support']:<8}")

    print("-" * 60)
    print(
        f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.4f} {report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f}")
    print(
        f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.4f} {report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f}")

    model.train()
    return accuracy


def train(args):
    model_name = args.model_path
    lr = args.learning_rate
    num_warmup_steps = args.num_warmup_steps
    epochs = args.epochs
    tb_writer = SummaryWriter(log_dir=args.log_dir)
    output_dir = args.output_dir
    batch_size = args.batch_size
    gradient_accumulation = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm
    log_step = args.log_step

    train_path = args.train_path
    val_path = args.val_path

    sen_label = readFile(os.path.join(train_path, 'sentence'), os.path.join(train_path, 'label'))

    num_batches = len(sen_label) // batch_size
    if len(sen_label) % batch_size != 0:
        num_batches += 1

    total_steps = (num_batches // gradient_accumulation) * epochs
    print(f"Total optimization steps: {total_steps} (Batches: {num_batches}, Gradient Accum: {gradient_accumulation})")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    target_modules = args.target_modules.split(",")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    print('starting training')
    overall_step = 0
    running_loss = 0
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.datetime.now()
        print('time: {}'.format(now))

        random.shuffle(sen_label)

        for step in range(0, len(sen_label), batch_size):
            batch = sen_label[step:step + batch_size]

            input_texts = []
            only_input_lens = []

            for input_text, target_label in batch:
                clean_input = preprocess(input_text)
                clean_target = preprocess(target_label)
                full_prompt = build_prompt(clean_input, clean_target)
                input_texts.append(full_prompt)

                input_only_prompt = build_prompt(clean_input, None)
                input_len = len(tokenizer.encode(input_only_prompt, add_special_tokens=True))
                only_input_lens.append(input_len)

            encodings = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask

            labels = input_ids.clone()

            for i, input_len in enumerate(only_input_lens):
                input_len = min(input_len, labels.size(1))
                labels[i, :input_len] = -100
                labels[attention_mask == 0] = -100

            labels = labels.to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if ((step // batch_size) + 1) % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                overall_step += 1

            running_loss += loss.item()

            if overall_step % log_step == 0:
                avg_loss = running_loss / log_step
                tb_writer.add_scalar('train_loss', avg_loss, overall_step)
                print('Epoch {}, Global Step {}, Avg Loss {:.4f}'.format(
                    epoch + 1, overall_step, avg_loss))
                running_loss = 0


        output_res_path = os.path.join(args.output_dir, "test_eval.jsonl")
        val_accuracy = evaluate(model, tokenizer, val_path, device, desc="Validation",
                                          batch_size=batch_size, output_file=output_res_path)
        tb_writer.add_scalar('val_accuracy', val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(os.path.join(output_dir, 'best_lora_model'))
            tokenizer.save_pretrained(os.path.join(output_dir, 'best_lora_model'))

        model.save_pretrained(os.path.join(output_dir, f'lora_model_epoch_{epoch + 1}'))

    model.save_pretrained(os.path.join(output_dir, 'final_lora_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'final_lora_model'))
    print('training finished')


def test(args):

    model_name = args.model_path
    test_path = args.test_path
    lora_model_path = os.path.join(args.output_dir, args.lora_model_name)


    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )


    model = PeftModel.from_pretrained(model, lora_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    output_res_path = os.path.join(args.output_dir, "test_predictions.jsonl")

    evaluate(model, tokenizer, test_path, device,
             desc="Test", batch_size=args.batch_size, output_file=output_res_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek model with LoRA")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation (default: 4)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--log_step", type=int, default=10, help="Log every n steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_model_name", type=str, default="final_lora_model",
                        help="Name of the LoRA model to load for testing")
    parser.add_argument("--prediction_file", type=str, default="predictions.jsonl",
                        help="Name of the file to save predictions")

    args = parser.parse_args()
    print("Parsed arguments:", args)

    train(args)
    print("Training completed. Starting test...")


if __name__ == "__main__":
    main()
