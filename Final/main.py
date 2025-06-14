import torch
import random
import numpy as np
from torchvision.models import resnet18
from wm_student import StudentWatermarkTask  # Implement your watermark injection

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    # Hyper parameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    num_models = 10
    perturbation_factor = 2e-3
    clean_model_path = "resnet18_cifar10_best.pth"

    # Load pretrained models
    clean_model = resnet18(pretrained=True)
    clean_model.fc = torch.nn.Linear(clean_model.fc.in_features, 10)
    checkpoint = torch.load(clean_model_path)
    clean_model.load_state_dict(checkpoint['model_state_dict'])
    clean_model = clean_model.to(device)

    # Initialize your watermark framework
    watermark_task = StudentWatermarkTask(
        clean_model=clean_model,
        num_models=num_models,
        perturbation_factor=perturbation_factor,
        batch_size=batch_size,
        device=device
    )

    # Start evaluate
    print("Evaluating watermarking task...")
    watermark_task.evaluate()

if __name__ == "__main__":
    main()
