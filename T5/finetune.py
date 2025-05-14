import os
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import numpy as np
import pandas as pd
import datetime

device = 'cuda' if cuda.is_available() else 'cpu'


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class
        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.rewrite_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            #padding=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.rewrite_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            #padding=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }

def readFile(sen_file, label_file):

    sen_label = []
    sens = []
    labs = []
    sen_label.append(["input", "label"])
    with open(sen_file,'r') as fr:
        for line in fr:
            sens.append(line.strip())
    with open(label_file,'r', encoding='utf-8', errors='ignore') as fr:
        for line in fr:
            labs.append(line.strip())
    for sen, lab in zip(sens,labs):
        sen_label.append([sen, lab])

    return sen_label

def train(epoch, tokenizer, model, device, loader, optimizer, summary_writer, output_dir):
    """
    Function to be called for training with the parameters passed from main function
    """

    model.train()
    for _, data in enumerate(loader, 0):
        #print (_)
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
             input_ids=ids,
             attention_mask = mask,
             labels=y,
         )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        summary_writer.add_scalar('epoch/loss_{}'.format(epoch), loss.item(), _)



def T5Trainer(
        dataframe, source_text, target_text, model_params, output_dir="./outputs/semeval/unifiedqa_v2_t5_large_1251000"
):
    """
    T5 trainer
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed

    # logging
    print(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    #tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"], quantization_config=bnb_config)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    print(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 1
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    train_dataset = train_dataset.reset_index(drop=True)
    #print(train_dataset)

    print(f"FULL Dataset: {dataframe.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    print(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        summary_writer = SummaryWriter(log_dir="t5/summary_task")
        train(epoch, tokenizer, model, device, training_loader, optimizer, summary_writer, output_dir)
        print(f"[Saving Model]...\n")
        # Saving the model after training
        path = os.path.join(output_dir, 'model_large_epoch_ori_notag_{}'.format(epoch + 1+4))
        if not os.path.exists(path):
            os.mkdir(path)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )


if __name__ == '__main__':
    model_params = {
        "MODEL": "allenai/unifiedqa-v2-t5-large-1251000",  # model_type: t5-base/t5-large


        "TRAIN_BATCH_SIZE": 4,  # training batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "LEARNING_RATE": 1.5e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 500,  # max length of source text  defi semeval 750   acl 1000  #semeval 329 acl 581 or 500;
        "MAX_TARGET_TEXT_LENGTH": 40,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }

        sen_label = readFile('train_file', 'test_file')
        train_dataframe = pd.DataFrame(sen_label[1:], columns=sen_label[0])
        output_dir = "output_file"
        T5Trainer(train_dataframe, "input", "label", model_params, output_dir=output_dir)