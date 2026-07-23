import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import argparse
import random


class RBERT_Independent(nn.Module):
    def __init__(self, model_path, num_labels):
        super(RBERT_Independent, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size


        self.cls_fc = nn.Linear(hidden_size, hidden_size)
        self.entity_fc = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.label_classifier = nn.Linear(hidden_size * 3, num_labels)

    def get_token_avg(self, input_ids, mask):

        outputs = self.bert(input_ids, attention_mask=mask)
        last_hidden = outputs.last_hidden_state

        mask_expanded = mask.unsqueeze(-1).expand_as(last_hidden)
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        num_tokens = torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-9)
        return sum_embeddings / num_tokens

    def forward(self, sent_ids, sent_mask, e1_ids, e1_mask, e2_ids, e2_mask):

        sent_outputs = self.bert(sent_ids, attention_mask=sent_mask)
        cls_vector = sent_outputs.last_hidden_state[:, 0, :]
        h0_prime = self.cls_fc(self.activation(cls_vector))


        e1_avg = self.get_token_avg(e1_ids, e1_mask)
        e2_avg = self.get_token_avg(e2_ids, e2_mask)

        h1_prime = self.entity_fc(self.activation(e1_avg))
        h2_prime = self.entity_fc(self.activation(e2_avg))


        combined = torch.cat([h0_prime, h1_prime, h2_prime], dim=-1)
        return self.label_classifier(self.dropout(combined))


def data_read(sentence_file, tag_file, entity_file, data_type='train'):
    IRRELEVANT_LABEL = "(J) no-relation"
    T_DIVISOR = 3
    data_list = []

    with open(sentence_file, 'r', encoding='utf-8') as f_sen, \
            open(tag_file, 'r', encoding='utf-8') as f_tag, \
            open(entity_file, 'r', encoding='utf-8') as f_ent:

        for sen, tag, ent in zip(f_sen, f_tag, f_ent):
            sen = sen.strip().replace(" ##", "").replace("##", "")
            ent_parts = ent.strip().split('\t')
            if len(ent_parts) < 2: continue

            tag_str = tag.strip().lower()

            data_list.append({
                "text": sen,
                "label": tag_str,
                "e1": ent_parts[0].strip(),
                "e2": ent_parts[1].strip()
            })

    if data_type == 'train':

        relevant_pairs_by_label = {}
        irrelevant_pairs = []

        for pair in data_list:
            label = pair['label']
            if label == IRRELEVANT_LABEL:
                irrelevant_pairs.append(pair)
            else:
                relevant_pairs_by_label.setdefault(label, []).append(pair)


        balanced_pairs = []
        N = len(irrelevant_pairs)

        for label, pairs in relevant_pairs_by_label.items():
            M_i = len(pairs)

            replication = max(1, int(N / (T_DIVISOR * M_i))) if M_i > 0 else 1
            for p in pairs:
                for _ in range(replication):
                    balanced_pairs.append(p)


        balanced_pairs.extend(irrelevant_pairs)
        random.shuffle(balanced_pairs)

        print(f"Original training data: {len(data_list)} samples | Balanced data: {len(balanced_pairs)} samples")
        return balanced_pairs
    else:
        print(f"Original {data_type}data: {len(data_list)} samples | Kept unchanged")
        return data_list


class RBERTDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128, ent_max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.ent_max_len = ent_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]


        s_enc = self.tokenizer(item['text'], max_length=self.max_len, padding='max_length', truncation=True,
                               return_tensors="pt")
        e1_enc = self.tokenizer(item['e1'], max_length=self.ent_max_len, padding='max_length', truncation=True,
                                return_tensors="pt")
        e2_enc = self.tokenizer(item['e2'], max_length=self.ent_max_len, padding='max_length', truncation=True,
                                return_tensors="pt")

        return {
            'sent_ids': s_enc['input_ids'].squeeze(),
            'sent_mask': s_enc['attention_mask'].squeeze(),
            'e1_ids': e1_enc['input_ids'].squeeze(),
            'e1_mask': e1_enc['attention_mask'].squeeze(),
            'e2_ids': e2_enc['input_ids'].squeeze(),
            'e2_mask': e2_enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.label2id.get(item['label'], 0))
        }


def evaluate(model, loader, device, id2label, desc="Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            labels = batch['label'].to(device)
            logits = model(
                batch['sent_ids'].to(device), batch['sent_mask'].to(device),
                batch['e1_ids'].to(device), batch['e1_mask'].to(device),
                batch['e2_ids'].to(device), batch['e2_mask'].to(device)
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    return total_loss / len(loader), micro_f1, all_labels, all_preds


def main(args):
    fold = args.fold
    MODEL_PATH = "google-bert/bert-base-uncased"
    BATCH_SIZE = 16
    LR = 2e-5
    EPOCHS = 5
    OUTPUT_DIR = f"../output/output_R_BERT/{fold}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


    base_data_path = f"../datasets/Abstract/SLM/{fold}"


    train_data = data_read(f"{base_data_path}/train/sentence", f"{base_data_path}/train/label",
                           f"{base_data_path}/train/entity", 'train')
    val_data = data_read(f"{base_data_path}/val/sentence", f"{base_data_path}/val/label", f"{base_data_path}/val/entity",
                         'val')
    test_data = data_read(f"{base_data_path}/val/sentence", f"{base_data_path}/val/label",
                          f"{base_data_path}/val/entity", 'test')

    label2id = {l: i for i, l in enumerate(sorted(list(set([d['label'] for d in train_data]))))}
    id2label = {v: k for k, v in label2id.items()}

    train_loader = DataLoader(RBERTDataset(train_data, tokenizer, label2id), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RBERTDataset(val_data, tokenizer, label2id), batch_size=BATCH_SIZE)
    test_loader = DataLoader(RBERTDataset(test_data, tokenizer, label2id), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RBERT_Independent(MODEL_PATH, len(label2id)).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(
                batch['sent_ids'].to(device), batch['sent_mask'].to(device),
                batch['e1_ids'].to(device), batch['e1_mask'].to(device),
                batch['e2_ids'].to(device), batch['e2_mask'].to(device)
            )
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        _, v_f1, _, _ = evaluate(model, val_loader, device, id2label, "Validating")
        print(f"Epoch {epoch + 1} | Loss: {train_loss / len(train_loader):.4f} | Val F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.bin")


    print("\n--- Final Test ---")
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.bin"))
    _, t_f1, y_true, y_pred = evaluate(model, test_loader, device, id2label, "Testing")

    output_file_path =  f"{OUTPUT_DIR}/test_results.txt"

    print(f"Test results successfully saved to: {output_file_path}")


    id2label = {v: k for k, v in label2id.items()}
    print(id2label)


    y_pred_labels = [id2label[idx] for idx in y_pred]

    with open(output_file_path, "w", encoding="utf-8") as f:
        for label in y_pred_labels:
            f.write(f"{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default="fold_0")
    main(parser.parse_args())