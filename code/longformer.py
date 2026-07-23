import os
import torch
import torch.nn as nn
import re
from torch.utils.data import DataLoader, Dataset
from transformers import LongformerModel, LongformerTokenizer, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import random
import argparse


def data_read(sentence_file, tag_file, entity_file, data_type='train'):
    IRRELEVANT_LABEL = "(J) no-relation"
    T_DIVISOR = 3
    data_list = []

    with open(sentence_file, 'r', encoding='utf-8') as f_sen, \
            open(tag_file, 'r', encoding='utf-8') as f_tag, \
            open(entity_file, 'r', encoding='utf-8') as f_ent:

        for sen, tag, ent in zip(f_sen, f_tag, f_ent):
            sen = sen.strip().replace(" ##", "").replace("##", "")
            tag = tag.strip().lower()
            ent_parts = ent.strip().split('\t')
            if len(ent_parts) < 2: continue

            data_list.append({
                "text": sen, "label": tag,
                "e1": ent_parts[0].strip(), "e2": ent_parts[1].strip()
            })

    if data_type == 'train':
        relevant_pairs_by_label = {}
        irrelevant_pairs = [p for p in data_list if p['label'] == IRRELEVANT_LABEL]
        for p in data_list:
            if p['label'] != IRRELEVANT_LABEL:
                relevant_pairs_by_label.setdefault(p['label'], []).append(p)

        balanced_pairs = []
        N = len(irrelevant_pairs)
        for label, pairs in relevant_pairs_by_label.items():
            M_i = len(pairs)
            replication = max(1, int(N / (T_DIVISOR * M_i))) if M_i > 0 else 1
            for p in pairs:
                for _ in range(replication): balanced_pairs.append(p)
        balanced_pairs.extend(irrelevant_pairs)
        random.shuffle(balanced_pairs)
        return balanced_pairs
    return data_list


class REDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=512, ent_max_len=64):
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
            'sent_ids': s_enc['input_ids'].squeeze(0), 'sent_mask': s_enc['attention_mask'].squeeze(0),
            'e1_ids': e1_enc['input_ids'].squeeze(0), 'e1_mask': e1_enc['attention_mask'].squeeze(0),
            'e2_ids': e2_enc['input_ids'].squeeze(0), 'e2_mask': e2_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label2id[item['label']], dtype=torch.long)
        }


class LongformerFullIndependentRE(nn.Module):
    def __init__(self, model_path, num_labels, lstm_hidden=256, trans_heads=8):
        super(LongformerFullIndependentRE, self).__init__()
        self.longformer = LongformerModel.from_pretrained(model_path)
        hidden_size = self.longformer.config.hidden_size


        self.sent_bilstm = nn.LSTM(hidden_size, lstm_hidden, bidirectional=True, batch_first=True)
        self.sent_transformer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden * 2,
            nhead=trans_heads,
            batch_first=True
        )


        self.entity_fc = nn.Linear(hidden_size, lstm_hidden * 2)
        self.activation = nn.Tanh()


        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 6, lstm_hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden * 2, num_labels)
        )

    def get_entity_repr(self, input_ids, mask):

        outputs = self.longformer(input_ids, attention_mask=mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = mask.unsqueeze(-1).expand_as(last_hidden)
        avg_pool = torch.sum(last_hidden * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        return self.activation(self.entity_fc(avg_pool))

    def forward(self, sent_ids, sent_mask, e1_ids, e1_mask, e2_ids, e2_mask):

        sent_output = self.longformer(sent_ids, attention_mask=sent_mask).last_hidden_state
        lstm_out, _ = self.sent_bilstm(sent_output)


        trans_out = self.sent_transformer(lstm_out)


        s_mask_expanded = sent_mask.unsqueeze(-1).expand_as(trans_out)
        sent_feature = torch.sum(trans_out * s_mask_expanded, 1) / torch.clamp(s_mask_expanded.sum(1), min=1e-9)


        e1_feature = self.get_entity_repr(e1_ids, e1_mask)
        e2_feature = self.get_entity_repr(e2_ids, e2_mask)


        combined = torch.cat([sent_feature, e1_feature, e2_feature], dim=-1)
        return self.classifier(combined)


def evaluate(model, loader, device, id2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            logits = model(
                batch['sent_ids'].to(device), batch['sent_mask'].to(device),
                batch['e1_ids'].to(device), batch['e1_mask'].to(device),
                batch['e2_ids'].to(device), batch['e2_mask'].to(device)
            )
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return f1_score(all_labels, all_preds, average='micro'), all_labels, all_preds


def main(args):
    fold = args.fold
    MODEL_PATH = "allenai/longformer-base-4096"
    BATCH_SIZE = 8
    LR = 2e-5

    OUTPUT_DIR = f"../output/output_longformer/{fold}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = LongformerTokenizer.from_pretrained(MODEL_PATH)
    data_dir = f"../datasets/Abstract/SLM/{fold}"

    train_data = data_read(f"{data_dir}/train/sentence", f"{data_dir}/train/label", f"{data_dir}/train/entity", 'train')
    val_data = data_read(f"{data_dir}/val/sentence", f"{data_dir}/val/label", f"{data_dir}/val/entity", 'val')

    label2id = {l: i for i, l in enumerate(sorted(list(set([d['label'] for d in train_data]))))}
    id2label = {i: l for l, i in label2id.items()}

    train_loader = DataLoader(REDataset(train_data, tokenizer, label2id), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(REDataset(val_data, tokenizer, label2id), batch_size=BATCH_SIZE)
    test_loader = DataLoader(REDataset(val_data, tokenizer, label2id), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LongformerFullIndependentRE(MODEL_PATH, len(label2id)).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(5):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(
                batch['sent_ids'].to(device), batch['sent_mask'].to(device),
                batch['e1_ids'].to(device), batch['e1_mask'].to(device),
                batch['e2_ids'].to(device), batch['e2_mask'].to(device)
            )
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()

        v_f1, _, _ = evaluate(model, val_loader, device, id2label)
        print(f"Epoch {epoch + 1} | Val F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.bin")


    print("\n--- Final Test ---")
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.bin"))
    t_f1, y_true, y_pred = evaluate(model, test_loader, device, id2label)

    output_file_path = f"{OUTPUT_DIR}/test_results.txt"

    print(f"Test results successfully saved to: {output_file_path}")


    id2label = {v: k for k, v in label2id.items()}
    print(id2label)
    target_names = [id2label[i] for i in range(len(label2id))]
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"Final Test Micro F1: {t_f1:.4f}")

    y_pred_labels = [id2label[idx] for idx in y_pred]

    with open(output_file_path, "w", encoding="utf-8") as f:
        for label in y_pred_labels:
            f.write(f"{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default="fold_0")
    main(parser.parse_args())