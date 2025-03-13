from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import copy
from .client import *
from tqdm import tqdm

class ClientSST(Client):
    def __init__(self, client_id, model, data, device):
        super().__init__(client_id, model, data, device)

    def train(self, epochs=5, batch_size=16):
        """Local training on client's data"""
        self.model.to(self.device)
        # dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(tqdm(self.data)):
            # for batch in tqdm(self.data):
                batch.to(self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()

    def evaluate(self, eval_dataloader):
        self.model.to(self.device)
        self.model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(self.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric

    def get_parameters(self):
        """Return model parameters for federated averaging"""
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, new_params):
        """Set model parameters received from the server"""
        self.model.load_state_dict(new_params)