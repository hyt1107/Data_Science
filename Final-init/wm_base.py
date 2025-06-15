import abc
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from typing import final
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18 as _resnet18

class WatermarkTask(abc.ABC):
    """
    Base class for watermark insertion, detector training, and evaluation.
    """
    
    def __init__(self, clean_model: nn.Module, num_models: int = 10, perturbation_factor: float = 0.0001, batch_size: int = 128, device: str = 'cpu'):
        """
        Initializes the WatermarkTask class.
        """
        self.device = torch.device(device)
        
        # Generate perturbed clean models
        print("Generating perturbed clean models...")
        self.clean_models = self.generate_clean_models(clean_model, num_models, perturbation_factor)

        # Load CIFAR-10 test set for evaluation
        print("    Loading CIFAR-10 test set...")
        transform = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), 
        ])
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Feature extraction: grab one batch of images for feature extraction
        print("    Extracting features from CIFAR-10 test set...")
        self.feature_imgs, _ = next(iter(self.test_loader))
        self.feature_imgs = self.feature_imgs.to(self.device)

        # Load CIFAR-10 training set for fine-tuning
        transform_train = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    def generate_clean_models(self, clean_model: nn.Module, num_models: int, perturbation_factor: float) -> list:
        """
        Generates clean models by applying perturbation to the original model.
        """
        print(f"    Generating {num_models} perturbed models with perturbation factor {perturbation_factor}...")
        
        def perturb_model(model: nn.Module, perturbation_factor: float) -> nn.Module:
            """
            Perturbs the model's parameters slightly by adding random noise.
            """
            perturbed_model = copy.deepcopy(model)
            with torch.no_grad():
                for param in perturbed_model.parameters():
                    if param.dim() > 1:  # Skip biases
                        noise = torch.randn_like(param) * perturbation_factor
                        param.add_(noise)
            return perturbed_model

        # Generate perturbed clean models
        return [perturb_model(clean_model, perturbation_factor) for _ in range(num_models)]

    @abc.abstractmethod
    def insert_watermark(self, model: nn.Module) -> nn.Module:
        """
        [TODO] You are aimed to implement this virtual function to
        insert a watermark into the given model. 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self, model: nn.Module) -> torch.Tensor:
        """
        05/20 Update
        [TODO] Extract custom features from the given model.
        These features will be used as input to the detector.        
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 100) -> nn.Module:
        """
        [TODO] You are aimed to implement this virtual function to train the watermark detector.
        Output: 1 if watermarked, 0 if clean.
        """
        raise NotImplementedError
    
    
    @final
    def evaluate_main_task(self, model: nn.Module) -> float:
        """
        Compute the top-1 accuracy of the model on the CIFAR-10 test set.
        """
        print(f"    Evaluating model accuracy on CIFAR-10 test set...")
        model = model.to(self.device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total * 100
        return accuracy

    @final
    def fine_tune_attack(self, model: nn.Module, lr: float = 0.01, epochs: int = 30) -> nn.Module:
        """
        Fine-tune attack on a watermarked model using CIFAR-10 training data.
        """
        print(f"    Performing fine-tune attack using CIFAR-10 training data...")
        attacked_model = copy.deepcopy(model).to(self.device).train()
        optimizer = optim.Adam(attacked_model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        # Fine-tune using CIFAR-10 training set
        for _ in range(epochs):
            attacked_model.train()
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = attacked_model(imgs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
        return attacked_model

    @final
    def prune_attack(self, model: nn.Module, prune_percentage: float = 0.50) -> nn.Module:
        """
        Prune (remove) a percentage of the smallest weights in the model.
        """
        print(f"    Performing pruning attack...")
        attacked_model = copy.deepcopy(model).to(self.device)
        with torch.no_grad():
            for name, param in attacked_model.named_parameters():
                if param.dim() in [2, 4]:  # Conv and Linear layers
                    threshold = torch.quantile(param.abs(), prune_percentage)
                    mask = param.abs() > threshold
                    param.mul_(mask.float())
        return attacked_model

    @final
    def evaluate_detection_rate(self, detector_model: nn.Module, clean_models: list, wm_models: list) -> float:
        """
        Evaluate the detection rate of the watermarking detector on given models (watermarked, fine-tuned, pruned).
        
        Args:
            detector_model (nn.Module): The trained watermark detector.
            clean_models (list): List of clean models.
            wm_models (list): List of watermarked models.
        
        Returns:
            float: Detection rate as a percentage.
        """
        correct = 0
        all_models = clean_models + wm_models  # Combine clean and watermarked models for evaluation

        for model in all_models:
            # Extract the features from the model and move them to the same device as the detector model
            features = self.extract_features(model).to(self.device)

            # Predict whether the model is watermarked using the detector model
            output = detector_model(features.unsqueeze(0))
            _, predicted = torch.max(output, dim=1)

            # Compare the predicted label with the actual label (0 for clean, 1 for watermarked)
            actual_label = 0 if model in clean_models else 1
            if predicted.item() == actual_label:
                correct += 1
        
        detection_rate = correct / len(all_models) * 100
        return detection_rate

    @final
    def evaluate(self):
        """
        Full evaluation pipeline:
        1. Insert watermarks into clean models.
        2. Train the watermark detector on clean vs watermarked models.
        3. Evaluate detection rate for watermarked, fine-tuned, and pruned models.
        4. Evaluate main task accuracy for clean, watermarked, fine-tuned, and pruned models.
        """
        # Inject watermark
        print("Inject watermark into clean model")
        wm_models = [self.insert_watermark(m) for m in self.clean_models]

        # Train detector
        print("Training watermark detector")
        detector_model = self.train_detector(self.clean_models, wm_models)

        # Fine tune and prune attack
        print("Attacking watermarked model")
        ft_models = [self.fine_tune_attack(m) for m in wm_models]
        pr_models = [self.prune_attack(m) for m in wm_models]

        # Evaulate detection rate
        print("Evaluate detection rate")
        detection_rate_wm = self.evaluate_detection_rate(detector_model, self.clean_models, wm_models)
        detection_rate_ft = self.evaluate_detection_rate(detector_model, self.clean_models, ft_models)
        detection_rate_pr = self.evaluate_detection_rate(detector_model, self.clean_models, pr_models)


        # Evaluate the accuracy
        print("Evaluate main task accuracy")
        acc_clean = sum(self.evaluate_main_task(m) for m in self.clean_models) / len(self.clean_models)
        acc_wm = sum(self.evaluate_main_task(m) for m in wm_models) / len(wm_models)
        acc_ft = sum(self.evaluate_main_task(m) for m in ft_models) / len(ft_models)
        acc_pr = sum(self.evaluate_main_task(m) for m in pr_models) / len(pr_models)


        # Print the evaluation results
        print("\nEvaluation results:")
        print(f"Main Task Accuracy (Clean Models): {acc_clean:.2f}%")
        print(f"Main Task Accuracy (Watermarked Models): {acc_wm:.2f}%")
        print(f"Main Task Accuracy (Fine-Tuned Models): {acc_ft:.2f}%")
        print(f"Main Task Accuracy (Pruned Models): {acc_pr:.2f}%")
        print(f"Detection Rate (Watermarked Models): {detection_rate_wm:.2f}%")
        print(f"Detection Rate (Fine-Tuned Models): {detection_rate_ft:.2f}%")
        print(f"Detection Rate (Pruned Models): {detection_rate_pr:.2f}%")
