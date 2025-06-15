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
        """
        Insert watermark into the model by adding small perturbations to the weights.
        """
        watermarked_model = copy.deepcopy(model)
        for param in watermarked_model.parameters():
            if param.requires_grad:
                # Generate random perturbation with the same shape as the parameter
                perturbation = torch.randn_like(param) * self.perturbation_factor
                # Add perturbation to the parameter
                param.data += perturbation
        return watermarked_model

    def extract_features(self, model: nn.Module) -> torch.Tensor:
        """
        Extract features from the model by concatenating all trainable parameters.
        """
        features = []
        for param in model.parameters():
            if param.requires_grad:
                # Flatten and normalize the parameter
                flat_param = param.data.flatten()
                normalized_param = (flat_param - flat_param.mean()) / (flat_param.std() + 1e-8)
                features.append(normalized_param)
        return torch.cat(features)

    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 100):
        """
        Train a detector to distinguish between clean and watermarked models.
        """
        class WatermarkDetector(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 2)
                )
            
            def forward(self, x):
                return self.network(x)

        # Extract features from all models
        clean_features = torch.stack([self.extract_features(model) for model in clean_models])
        wm_features = torch.stack([self.extract_features(model) for model in wm_models])
        
        # Create labels (0 for clean, 1 for watermarked)
        clean_labels = torch.zeros(len(clean_models))
        wm_labels = torch.ones(len(wm_models))
        
        # Combine features and labels
        X = torch.cat([clean_features, wm_features])
        y = torch.cat([clean_labels, wm_labels]).long()
        
        # Initialize detector
        input_size = X.shape[1]
        detector = WatermarkDetector(input_size).to(self.device)
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(detector.parameters(), lr=0.001)
        
        # Training loop
        detector.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = detector(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        return detector
