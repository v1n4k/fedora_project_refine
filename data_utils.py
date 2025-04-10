# data_utils.py
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import evaluate
import torch

def load_glue_dataset(dataset_name, model_name_or_path="roberta-large", num_clients=3, batch_size=32):
    """Tải dataset từ GLUE."""
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", dataset_name)
    metric = evaluate.load("glue", dataset_name)
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer, dataset_name=dataset_name),
        batched=True,
        remove_columns=["idx", "sentence"] if dataset_name == "sst2" else ["idx", "question1", "question2"] if dataset_name == "qqp" else 
                        ["idx", "question", "sentence"] if dataset_name == "qnli" else ["idx", "premise", "hypothesis"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    if dataset_name == "mnli_matched" or dataset_name == "mnli_mismatched":
        # train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
        eval_dataloader = DataLoader(tokenized_datasets["validation_matched"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
        mismatched_eval_dataloader = DataLoader(tokenized_datasets["validation_mismatched"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
    else:
        # train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
        eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    total_len = len(tokenized_datasets["train"])
    split_size = total_len // num_clients
    remainder = total_len % num_clients
    split_lengths = [split_size] * num_clients
    for i in range(remainder):
        split_lengths[i] += 1
    train_dataloader = torch.utils.data.random_split(tokenized_datasets["train"], split_lengths)
    client_data_splits = [DataLoader(train_dataloader[i], shuffle=True, collate_fn=collate_fn, batch_size=batch_size) for i in range(num_clients)]

    if dataset_name == "mnli_mismatched":
        return client_data_splits, eval_dataloader, mismatched_eval_dataloader, metric
    else:
        return client_data_splits, eval_dataloader, metric

def split_non_iid(dataset, num_clients=3):
    """Chia dữ liệu thành non-IID cho các client."""
    data_size = len(dataset)
    indices = np.random.permutation(data_size)
    client_size = data_size // num_clients
    client_data = []
    for i in range(num_clients):
        start = i * client_size
        end = (i + 1) * client_size
        subset = dataset.select(indices[start:end])
        client_data.append(subset)
    return client_data

def tokenize_function(examples, tokenizer, dataset_name):
    # max_length=None => use the model max length (it's actually the default)
    if dataset_name=="sst2":
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    elif dataset_name=="qnli":
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=None)
    elif dataset_name=="qqp":
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=None)
    else:
        outputs = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=None)
    return outputs

def collate_fn(examples, tokenizer):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def split_data():
    pass
