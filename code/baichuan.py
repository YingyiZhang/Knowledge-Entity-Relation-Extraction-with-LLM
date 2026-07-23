import datetime
import json
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import classification_report, f1_score
import transformers


IRRELEVANT_LABEL = "(J) no-relation"
T_DIVISOR = 3


def preprocess(text):
    return text.replace("\n", "\\n").replace("\t", "\\t")


def build_baichuan_prompt(input_text, target_label=None):


    instruction = "Classify the relation of the following text:"
    input_part = f"### Input:\n{instruction} {preprocess(input_text)}"
    output_part = f"### Output:\n{preprocess(target_label)}" if target_label is not None else "### Output:"


    return f"{input_part}\n\n{output_part}"


def readFile(sen_file, label_file):
    sens, labs = [], []
    with open(sen_file, 'r', encoding='utf-8') as fr:
        for line in fr: sens.append(line.strip())
    with open(label_file, 'r', encoding='utf-8') as fr:
        for line in fr: labs.append(line.strip())

    original_pairs = [(s, l) for s, l in zip(sens, labs)]
    relevant_pairs_by_label, irrelevant_pairs = {}, []

    for pair in original_pairs:
        label = pair[1].strip().lower()
        if label == IRRELEVANT_LABEL.lower():
            irrelevant_pairs.append(pair)
        else:
            relevant_pairs_by_label.setdefault(label, []).append(pair)

    balanced_pairs = []
    N = len(irrelevant_pairs)
    for label, pairs in relevant_pairs_by_label.items():
        M_i = len(pairs)
        replication = max(0, int(N / (T_DIVISOR * M_i))) if M_i > 0 else 0
        print(f"Label '{label}': Replication times: {replication}")
        for p in pairs:
            for _ in range(replication): balanced_pairs.append(p)
    balanced_pairs.extend(irrelevant_pairs)
    print(f"Final dataset size: {len(balanced_pairs)}")
    return balanced_pairs


def evaluate(model, tokenizer, data_path, device, desc="Validation", batch_size=1, output_file=None):
    print(f"Starting {desc}...")

    model.eval()

    sen_file, label_file = os.path.join(data_path, 'sentence_explain'), os.path.join(data_path, 'label_prompt')
    eval_data = []
    with open(sen_file, 'r', encoding='utf-8') as fs, open(label_file, 'r', encoding='utf-8') as fl:
        for s, l in zip(fs, fl): eval_data.append((s.strip(), l.strip()))

    predictions, true_labels, results_to_save = [], [], []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i: i + batch_size]
            prompts = [build_baichuan_prompt(b[0]) for b in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            input_len = inputs.input_ids.shape[1]
            decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            for pred_text, (orig_s, orig_l) in zip(decoded, batch):
                p, t = pred_text.strip().lower(), orig_l.lower()
                predictions.append(p)
                true_labels.append(t)
                results_to_save.append({"input": orig_s, "true_label": t, "prediction": p})

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results_to_save: f.write(json.dumps(item, ensure_ascii=False) + "\n")

    unique_labels = sorted(list(set(true_labels + predictions)))
    macro_f1 = f1_score(true_labels, predictions, labels=unique_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, labels=unique_labels, average='micro', zero_division=0)


    exact_matches = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    accuracy = exact_matches / len(eval_data)

    report = classification_report(
        true_labels,
        predictions,
        labels=unique_labels,
        target_names=unique_labels,
        zero_division=0,
        output_dict=True
    )


    print(f"\n{desc} Results:")
    print(f"Accuracy: {accuracy:.4f}")
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
    return report['accuracy']


def train(args):

    tb_writer = SummaryWriter(log_dir=args.log_dir)


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    num_warmup_steps = args.num_warmup_steps
    gradient_accumulation = args.gradient_accumulation_steps
    batch_size = args.batch_size

    train_path = args.train_path
    val_path = args.val_path

    epochs = args.epochs

    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["W_pack"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    sen_label = readFile(os.path.join(train_path, 'sentence_explain'), os.path.join(train_path, 'label_prompt'))


    num_batches = len(sen_label) // batch_size
    if len(sen_label) % batch_size != 0:
        num_batches += 1
    total_steps = (num_batches // gradient_accumulation) * epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    overall_step = 0
    running_loss = 0.0
    best_val_accuracy = 0.0

    print(f'Starting training... Total steps: {total_steps}')
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(sen_label)

        for step in range(0, len(sen_label), args.batch_size):
            batch = sen_label[step: step + args.batch_size]
            input_ids_list, labels_list = [], []

            for s, l in batch:
                full_txt = build_baichuan_prompt(s, l) + tokenizer.eos_token
                prompt_txt = build_baichuan_prompt(s)

                full_ids = tokenizer.encode(full_txt)
                prompt_ids = tokenizer.encode(prompt_txt)

                input_ids_list.append(torch.tensor(full_ids))
                label = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
                labels_list.append(torch.tensor(label))

            inputs = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True,
                                                     padding_value=tokenizer.pad_token_id).to(model.device)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100).to(model.device)

            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item()

            if ((step // args.batch_size) + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                overall_step += 1


                if overall_step % args.log_step == 0:
                    avg_loss = (running_loss * args.gradient_accumulation_steps) / args.log_step
                    print(f"Epoch [{epoch + 1}/{args.epochs}], Step [{overall_step}], Loss: {avg_loss:.4f}")
                    tb_writer.add_scalar('train/loss', avg_loss, overall_step)
                    running_loss = 0.0


        output_res_path = os.path.join(args.output_dir, "test_eval.jsonl")
        val_accuracy = evaluate(model, tokenizer, val_path, model.device, batch_size=args.batch_size, output_file=output_res_path)
        tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(args.output_dir, "best_lora_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"--- Best model saved with accuracy: {best_val_accuracy:.4f} ---")


    final_save_path = os.path.join(args.output_dir, "final_lora_model")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f'Training finished. Final model saved to {final_save_path}')


def test(args):
    print("Starting Test Process...")
    model_name = args.model_path
    test_path = args.test_path

    lora_model_path = os.path.join(args.output_dir, args.lora_model_name)


    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


    print(f"Loading LoRA weights from {lora_model_path}...")
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()


    output_res_path = os.path.join(args.output_dir, "test_predictions.jsonl")


    evaluate(
        model,
        tokenizer,
        test_path,
        model.device,
        desc="Test Set",
        batch_size=args.batch_size,
        output_file=output_res_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)


    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)


    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)


    parser.add_argument("--lora_model_name", type=str, default="final_lora_model",
                        help="Name of the LoRA model directory to load after training")


    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--log_step", type=int, default=10)

    args = parser.parse_args()


    train(args)


if __name__ == "__main__":
    main()