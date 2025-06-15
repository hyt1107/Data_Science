import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

from wm_base import WatermarkTask

class StudentWatermarkTask(WatermarkTask):
    def __init__(self, clean_model: nn.Module, num_models: int = 10, perturbation_factor: float = 0.01, batch_size: int = 128, device: str = 'cuda'):
        """
        Initialize the student watermark task. 
        This class inherits from the base class `WatermarkTask` and implements the necessary methods.

        Args:
            clean_model (nn.Module): The clean pre-trained model.
            num_models (int): Number of perturbed models to generate.
            perturbation_factor (float): Factor to control the amount of perturbation.
            batch_size (int): Batch size for the CIFAR-10 DataLoader.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        super().__init__(clean_model, num_models, perturbation_factor, batch_size, device)

    def insert_watermark(self, model: nn.Module) -> nn.Module:
        #[TODO]
        wm_model = copy.deepcopy(model)
        wm_model.train()
        optimizer = torch.optim.Adam(wm_model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        trigger_imgs = self.feature_imgs[:10]  # trigger set
        trigger_labels = torch.zeros(10, dtype=torch.long).to(self.device)  # 都標為0類

        for i, (imgs, labels) in enumerate(self.train_loader):
            if i >= 3:  # 只做3個batch即可
                break
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            # 混合trigger和正常資料
            batch_imgs = torch.cat([imgs, trigger_imgs])
            batch_labels = torch.cat([labels, trigger_labels])
            optimizer.zero_grad()
            out = wm_model(batch_imgs)
            loss = loss_fn(out, batch_labels)
            loss.backward()
            optimizer.step()
        return wm_model

    def extract_features(self, model: nn.Module) -> torch.Tensor:
        #[TODO]
        model.eval()
        with torch.no_grad():
            trigger_imgs = self.feature_imgs[:10]
            logits = model(trigger_imgs)
            features = logits.flatten()
        return features
        # params = list(model.parameters())[0]
        # return params.flatten().mean().unsqueeze(0)

    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 100):
        #[TODO]
        """
        05/20 Update: Use extract_features to encode model information and assign label to clean model and watermark model
        , then train a detector on those features and label.
        """
        features = []
        labels = []
        for m in clean_models:
            features.append(self.extract_features(m))
            labels.append(0)
        for m in wm_models:
            features.append(self.extract_features(m))
            labels.append(1)
        X = torch.stack(features)
        y = torch.tensor(labels, dtype=torch.long)

        model = nn.Linear(X.shape[1], 2).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            #loss = loss_fn(output, y)
            loss = loss_fn(output, y.to(self.device))
            loss.backward()
            optimizer.step()
        return model
        
