import argparse
import os
from data_utils import load_glue_dataset, split_non_iid
from client import Client
from server import Server
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
import wandb
from datasets import load_dataset
import torch
from torch.optim import AdamW
import copy
import torch.distributed as dist
import torch.multiprocessing as mp

def main(args):
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    client_data_splits, eval_dataloader, metric = load_glue_dataset(dataset_name=args.dataset, num_clients=args.num_clients, batch_size=32)
    # client_data = split_non_iid(train_data, args.num_clients)
    peft_config = LoraConfig(use_dora=True, task_type="SEQ_CLS", inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, return_dict=True)
    model = get_peft_model(base_model, peft_config)

    server = Server(global_model=model, device=device)
    clients = [Client(client_id=i, model=copy.deepcopy(model), data=client_data_splits[i], device=device, train_method=args.method) for i in range(args.num_clients)]

    # wandb.init(project="FeDORA Project - QQP", entity="quanla", name="DoRA")

    with open(args.output_file, "w") as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of Clients: {args.num_clients}\n")
        f.write(f"Number of Rounds: {args.num_rounds}\n")
        f.write(f"Number of Epochs per Round: {args.num_epochs}\n\n")

        max_acc = 0
        for round_num in range(args.num_rounds):
            print(f"\n--- Federated Learning Round {round_num + 1} ---")
    
            client_models = []
            if "muon" in args.method:
                processes = []
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
            # Train each client locally
            for client in clients:
                print(f"\nTraining Client {client.client_id}")
                if "kd" in args.method:
                    client.train(epochs=args.num_epochs, train_method=args.method, base_model=base_model)
                else:
                    client.train(epochs=args.num_epochs, train_method=args.method)
                client_models.append(client.get_parameters())

            # Aggregate client models on the server
            print("\nAggregating client models on the server...")
            server.aggregate(client_models)
            eval_metric = server.evaluate(eval_dataloader, metric=metric)
            print(eval_metric)
            # wandb.log(eval_metric)
            # evaluate(lora_model, val_dataset)
    
            # Update each client's model with the new global model
            if max_acc<eval_metric['accuracy']:
                max_acc = eval_metric['accuracy']

            for client in clients:
                client.set_parameters(server.global_model.state_dict())

            f.write(f"  Global Accuracy after Round {round_num + 1}: {eval_metric['accuracy']:.4f}%\n\n")
        f.write(f"  Global Accuracy final: {eval_metric['accuracy']:.4f} and max_acc: {max_acc:.4f}%\n\n")
    print(f"Completed {args.method} on {args.dataset} with {args.model_name}. Final Accuracy: {eval_metric['accuracy']:.4f}%")
    # wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with DoRA/LoRA")
    parser.add_argument("--method", type=str, required=True, choices=["base", "fedora", "kd", "muon", "ns", "ns_manifold", "fedora+kd", "fedora+muon",
                        "fedora+kd+muon", "kd+muon", "muon+ns", "muon+ns_manifold"])
    parser.add_argument("--dataset", type=str, required=True, choices=["sst2", "qqp", "qnli", "mnli_matched", "mnli_mismatched"])
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model name from Hugging Face")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    args = parser.parse_args()
    main(args)