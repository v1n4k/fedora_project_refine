from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import copy
from .clientSST import *
from tqdm import tqdm

class ClientFeDoRA(ClientSST):
    def __init__(self, client_id, model, data, device):
        super().__init__(client_id, model, data, device)

    def train(self, epochs=5, batch_size=16, regularization_strength=1):
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

                is_done = False
                regu_loss = 0
                num_param = 0
                # cos = nn.CosineSimilarity(eps=1e-6)
                for n, p in self.model.named_parameters():
                    if 'base_layer.weight' in n:
                        base = p
                    if 'lora_A' in n:
                        lora_A = p
                    if 'lora_B' in n:
                        lora_B = p
                        is_done = True

                    if is_done:
                        # cos_loss = cos(base, lora_B@lora_A)
                        # regu_loss += cos_loss.norm()
                        sqr_matrix = (base+lora_B@lora_A).T@(base+lora_B@lora_A)
                        temp = torch.norm(torch.eye(sqr_matrix.shape[0]).cuda()-sqr_matrix)/sqr_matrix.shape[0]**1.5
                        regu_loss += temp

                        num_param += 1
                        is_done = False
                    
                # if num_param > 0:
                #     regu_loss = regu_loss / num_param
                # else:
                #     regu_loss = 0
                # print('loss:', loss, "regu loss:", regu_loss)
                loss += regularization_strength*regu_loss

                loss.backward()
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()

    # def evaluate(self, eval_dataloader):
    #     self.model.to(self.device)
    #     self.model.eval()
    #     for step, batch in enumerate(tqdm(eval_dataloader)):
    #         batch.to(self.device)
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1)
    #         predictions, references = predictions, batch["labels"]
    #         metric.add_batch(
    #             predictions=predictions,
    #             references=references,
    #         )

    #     eval_metric = metric.compute()
    #     return eval_metric

    # def get_parameters(self):
    #     """Return model parameters for federated averaging"""
    #     return copy.deepcopy(self.model.state_dict())

    # def set_parameters(self, new_params):
    #     """Set model parameters received from the server"""
    #     self.model.load_state_dict(new_params)