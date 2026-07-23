import datetime
import json
import os
import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import classification_report, f1_score
import transformers


IRRELEVANT_LABEL = "(J) no-relation"
T_DIVISOR = 3


def readFile(sen_file, label_file):

    sens = []
    labs = []
    with open(sen_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            sens.append(line.strip())
    with open(label_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            line=line.strip()
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
    total_loss = 0
    predictions = []
    true_labels = []
    results_to_save = []

    with torch.no_grad():

        for step in range(0, len(eval_data), batch_size):
            batch = eval_data[step:step + batch_size]
            input_texts = []
            target_texts = []
            batch_true_labels = []

            for input_text, target_label in batch:
                input_texts.append(preprocess(input_text))
                target_texts.append(postprocess(target_label))
                batch_true_labels.append(target_label.strip().lower())

            input_dict = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            labels_dict = tokenizer(
                target_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )

            labels = labels_dict.input_ids.to(device)


            outputs = model(
                input_ids=input_dict.input_ids,
                attention_mask=input_dict.attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item() * len(batch)


            generated_ids = model.generate(
                input_ids=input_dict.input_ids,
                attention_mask=input_dict.attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )


            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_preds = [pred.strip().lower() for pred in batch_preds]

            predictions.extend(batch_preds)
            true_labels.extend(batch_true_labels)

            for pred_text, orig_s, orig_l in zip(batch_preds, input_texts, batch_true_labels):
                p, t = pred_text.strip().lower(), orig_l.lower()
                predictions.append(p)
                true_labels.append(t)
                results_to_save.append({"input": orig_s, "true_label": t, "prediction": p})

            if step % (batch_size * 10) == 0:
                print(f"{desc} Step {step}/{len(eval_data)}")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results_to_save: f.write(json.dumps(item, ensure_ascii=False) + "\n")


    exact_matches = sum([1 for p, t in zip(predictions, true_labels) if p == t])
    accuracy = exact_matches / len(eval_data)


    unique_labels = sorted(list(set(true_labels)))

    report = classification_report(
        true_labels,
        predictions,
        labels=unique_labels,
        target_names=unique_labels,
        zero_division=0,
        output_dict=True
    )


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

    print(f"\n{desc} Results:")
    print(f"Accuracy: {accuracy:.4f}")

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

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",")
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
            target_texts = []

            for input_text, target_label in batch:
                input_texts.append(preprocess(input_text))
                target_texts.append(postprocess(target_label))


            input_dict = tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            labels_dict = tokenizer(
                target_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )
            labels = labels_dict.input_ids.to(device)


            outputs = model(
                input_ids=input_dict.input_ids,
                attention_mask=input_dict.attention_mask,
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


        output_res_path = os.path.join(args.output_dir, "test_oritag.jsonl")
        val_accuracy = evaluate(model, tokenizer, val_path, device, desc="Validation", batch_size=batch_size, output_file=output_res_path)
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

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, lora_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    output_res_path = os.path.join(args.output_dir, "test_predictions.jsonl")
    evaluate(model, tokenizer, test_path, device, desc="Test", batch_size=args.batch_size, output_file=output_res_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5 model with LoRA")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation (default: 4)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./t5_logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q,v")
    parser.add_argument("--lora_model_name", type=str, default="final_lora_model")

    args = parser.parse_args()
    print("Parsed arguments:", args)

    train(args)
    print("Training completed. Starting test...")


if __name__ == "__main__":
    main()