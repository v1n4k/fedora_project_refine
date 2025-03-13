from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import copy
from abc import ABC, abstractmethod
from opacus import PrivacyEngine
from torch import nn

class Client(ABC):
    def __init__(self, client_id, model, data, device, privacy_engine=None, num_epochs=5):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.data = data
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.privacy_engine = privacy_engine
        self.num_epochs = num_epochs

        # Privacy setup using make_private_with_epsilon (setting epsilon = 6)
        if self.privacy_engine is not None:
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=DataLoader(self.data, batch_size=16, shuffle=True),
                epochs=5,  # Number of epochs for which privacy accounting is done
                target_epsilon=6.0,  # Desired privacy budget
                target_delta=1e-5,
                max_grad_norm=1.0
            )

    @abstractmethod
    def train(self):
        pass